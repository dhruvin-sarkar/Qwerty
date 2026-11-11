//! Capture-device enumeration, RAW support and notifications (p2 §2.4).
//!
//! One way to do things (RUST-RT-28): enumeration, states, names and the RAW property use our
//! `windows` crate; notifications use wasapi's registration; the stream uses wasapi (capture.rs).
//! RAW support has one implementation, [`raw_processing_supported`], shared by listing and capture.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam_channel::{Sender, TrySendError};
use wasapi::{DeviceEnumerator, DeviceEventCallbacks, Direction, Role};
use windows::core::{Error as WinError, HRESULT, PWSTR};
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::{ERROR_NOT_FOUND, ERROR_NO_UNICODE_TRANSLATION, PROPERTYKEY};
use windows::Win32::Media::Audio::{
    eCapture, eConsole, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, DEVICE_STATE,
    DEVICE_STATEMASK_ALL, DEVICE_STATE_ACTIVE, DEVICE_STATE_DISABLED, DEVICE_STATE_NOTPRESENT,
    DEVICE_STATE_UNPLUGGED,
};
use windows::Win32::Storage::EnhancedStorage::PKEY_Devices_AudioDevice_RawProcessingSupported;
use windows::Win32::System::Com::StructuredStorage::{
    PropVariantClear, PropVariantToStringAlloc, PROPVARIANT,
};
use windows::Win32::System::Com::{CoCreateInstance, CoTaskMemFree, CLSCTX_ALL, STGM_READ};
use windows::Win32::System::Variant::VT_BOOL;

use crate::engine::capture::{self, map_wasapi, ComApartment};
use crate::engine::messages::{DeviceMsg, DeviceWatch};
use crate::engine::protocol::{CaptureError, CaptureStage, DeviceInfo, DeviceState, ProbeOutcome};

/// GetDefaultAudioEndpoint's "no endpoint of this flow is Active": HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
/// 0x80070490. (The `E_NOTFOUND` the windows crate exports is HtmlHelp's 0x8000100D, a different code.)
const E_NOTFOUND: HRESULT = ERROR_NOT_FOUND.to_hresult();

/// Controller thread (COM MTA). Our own IMMDeviceEnumerator::EnumAudioEndpoints(eCapture,
/// DEVICE_STATEMASK_ALL) (wasapi's collection lists ACTIVE only), friendly names from
/// PKEY_Device_FriendlyName, the default from GetDefaultAudioEndpoint(eCapture, eConsole) through
/// [`default_endpoint_id`], and a probe of each Active device. One device's probe failure is that
/// device's `ProbeOutcome::Failed` and never fails the listing. `Err` is only an enumerator-level
/// failure (ComInit, EnumAudioEndpoints, a GetDefaultAudioEndpoint failure other than E_NOTFOUND).
pub fn list_capture_devices() -> Result<Vec<DeviceInfo>, CaptureError> {
    // Balanced with the caller's own MTA init (S_FALSE is success and is uninitialized too), so the
    // listing also works from a thread that never initialized COM (the CLI).
    // SPEC: §2.4 lists ComInit among the listing's errors, so the listing enters COM itself.
    let _com = ComApartment::enter()?;
    // SAFETY: plain COM activation of the system enumerator on an MTA thread.
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
            .map_err(|e| os(CaptureStage::Enumerate, &e))?;
    // SAFETY: a live enumerator; the mask is the documented DEVICE_STATEMASK_ALL.
    let collection =
        unsafe { enumerator.EnumAudioEndpoints(eCapture, DEVICE_STATE(DEVICE_STATEMASK_ALL)) }
            .map_err(|e| os(CaptureStage::Enumerate, &e))?;
    let default = default_endpoint_id(default_capture_id(&enumerator))?;
    // SAFETY: a live collection.
    let count = unsafe { collection.GetCount() }.map_err(|e| os(CaptureStage::Enumerate, &e))?;
    let mut devices = Vec::with_capacity(count as usize);
    for index in 0..count {
        // SPEC: §2.4 makes only a device's probe per-device; a failure to read an endpoint's id,
        // state or name (which the listing cannot show without) is taken as enumerator-level.
        // SAFETY: index < count of a live collection.
        let device =
            unsafe { collection.Item(index) }.map_err(|e| os(CaptureStage::Enumerate, &e))?;
        let endpoint_id = endpoint_id(&device).map_err(|e| os(CaptureStage::Enumerate, &e))?;
        // SAFETY: a live device.
        let raw_state =
            unsafe { device.GetState() }.map_err(|e| os(CaptureStage::Enumerate, &e))?;
        let state = device_state(raw_state)?;
        let name = friendly_name(&device)?;
        let probe = match state {
            DeviceState::Active => match capture::probe(&endpoint_id) {
                Ok(probe) => ProbeOutcome::Probed { probe },
                Err(error) => ProbeOutcome::Failed { error },
            },
            DeviceState::Disabled | DeviceState::NotPresent | DeviceState::Unplugged => {
                ProbeOutcome::NotActive
            }
        };
        let is_default = default.as_deref() == Some(endpoint_id.as_str());
        devices.push(DeviceInfo {
            endpoint_id,
            name,
            is_default,
            state,
            probe,
        });
    }
    Ok(devices)
}

/// GetDefaultAudioEndpoint's result → the default's id. E_NOTFOUND (0x80070490) means no capture
/// endpoint is Active: Ok(None), every `is_default` false, and the listing still succeeds (it may
/// be empty or hold only non-Active devices). Any other failure is Err(Os{stage: Enumerate,
/// hresult}).
pub fn default_endpoint_id(
    result: windows::core::Result<String>,
) -> Result<Option<String>, CaptureError> {
    match result {
        Ok(id) => Ok(Some(id)),
        Err(e) if e.code() == E_NOTFOUND => Ok(None),
        Err(e) => Err(os(CaptureStage::Enumerate, &e)),
    }
}

/// OpenPropertyStore(STGM_READ) → PKEY_Devices_AudioDevice_RawProcessingSupported. A missing or
/// non-VT_BOOL value is Ok(false) (drivers without RAW support do not set it); a failing call is
/// Err(Os{stage: Properties, ..}).
pub fn raw_processing_supported(device: &IMMDevice) -> Result<bool, CaptureError> {
    read_property(
        device,
        &PKEY_Devices_AudioDevice_RawProcessingSupported,
        |value| {
            // SAFETY: `vt` is the discriminant every PROPVARIANT carries; `boolVal` is read only
            // when it says VT_BOOL.
            unsafe {
                let tagged = &value.Anonymous.Anonymous;
                tagged.vt == VT_BOOL && tagged.Anonymous.boolVal.0 != 0
            }
        },
    )
    .map_err(|e| os(CaptureStage::Properties, &e))
}

/// wasapi::DeviceEnumerator::register_notification_callback. Every closure is
/// `move |args..| forward_guarded(&dropped, || /* map args → DeviceMsg; the default_device filter
/// returns Ok(()) when not Capture/Console */ tx.try_send(msg))`, so the mapping and filtering also
/// run inside the guard.
pub fn watch(tx: Sender<DeviceMsg>, dropped: Arc<AtomicU64>) -> Result<DeviceWatch, CaptureError> {
    let enumerator = DeviceEnumerator::new().map_err(|e| map_wasapi(CaptureStage::Enumerate, e))?;
    let mut callbacks = DeviceEventCallbacks::new();
    {
        let (tx, dropped) = (tx.clone(), Arc::clone(&dropped));
        callbacks.set_device_added_callback(move |id| {
            forward_guarded(&dropped, || tx.try_send(DeviceMsg::Added { id }))
        });
    }
    {
        let (tx, dropped) = (tx.clone(), Arc::clone(&dropped));
        callbacks.set_device_removed_callback(move |id| {
            forward_guarded(&dropped, || tx.try_send(DeviceMsg::Removed { id }))
        });
    }
    {
        let (tx, dropped) = (tx.clone(), Arc::clone(&dropped));
        callbacks.set_device_state_callback(move |id, state| {
            forward_guarded(&dropped, || {
                tx.try_send(DeviceMsg::StateChanged {
                    id,
                    state: from_wasapi_state(state),
                })
            })
        });
    }
    {
        let (tx, dropped) = (tx.clone(), Arc::clone(&dropped));
        callbacks.set_default_device_callback(move |direction, role, id| {
            forward_guarded(&dropped, || match (direction, role) {
                (Direction::Capture, Role::Console) => {
                    tx.try_send(DeviceMsg::DefaultChanged { id })
                }
                (Direction::Capture, Role::Multimedia | Role::Communications)
                | (Direction::Render, Role::Console | Role::Multimedia | Role::Communications) => {
                    Ok(())
                }
            })
        });
    }
    callbacks.set_property_value_callback(move |id, _key| {
        forward_guarded(&dropped, || tx.try_send(DeviceMsg::PropertyChanged { id }))
    });
    let registration = enumerator
        .register_notification_callback(callbacks)
        .map_err(|e| map_wasapi(CaptureStage::Enumerate, e))?;
    Ok(DeviceWatch::new(Box::new(registration)))
}

/// R13: the only body of every wasapi device-notification closure (wasapi does not catch panics in
/// its COM callbacks). Runs `send` (message construction + filter + try_send) under
/// catch_unwind(AssertUnwindSafe(..)); increments `dropped` once on Err(_) or on a caught panic.
/// Never unwinds.
pub(crate) fn forward_guarded(
    dropped: &AtomicU64,
    send: impl FnOnce() -> Result<(), TrySendError<DeviceMsg>>,
) {
    match catch_unwind(AssertUnwindSafe(send)) {
        Ok(Ok(())) => {}
        Ok(Err(TrySendError::Full(_) | TrySendError::Disconnected(_))) => {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
        Err(payload) => {
            dropped.fetch_add(1, Ordering::Relaxed);
            // A payload whose Drop panics would unwind out of here into wasapi's COM callback;
            // leaking the payload is the one way to never unwind.
            std::mem::forget(payload);
        }
    }
}

/// wasapi's notification state → ours.
pub(crate) fn from_wasapi_state(state: wasapi::DeviceState) -> DeviceState {
    match state {
        wasapi::DeviceState::Active => DeviceState::Active,
        wasapi::DeviceState::Disabled => DeviceState::Disabled,
        wasapi::DeviceState::NotPresent => DeviceState::NotPresent,
        wasapi::DeviceState::Unplugged => DeviceState::Unplugged,
    }
}

/// The endpoint's friendly name (PKEY_Device_FriendlyName), for display only.
pub(crate) fn friendly_name(device: &IMMDevice) -> Result<String, CaptureError> {
    read_property(device, &PKEY_Device_FriendlyName, |value| {
        // SAFETY: PropVariantToStringAlloc accepts any PROPVARIANT and returns a CoTaskMem string
        // that take_string frees.
        unsafe { PropVariantToStringAlloc(value) }.map(|text| {
            // Display only and never compared (DeviceIdentity.name), so an unpaired surrogate is
            // shown as U+FFFD rather than hiding the device.
            // SAFETY: a valid, NUL-terminated CoTaskMem string from PropVariantToStringAlloc.
            let name = String::from_utf16_lossy(unsafe { text.as_wide() });
            // SAFETY: allocated by PropVariantToStringAlloc with CoTaskMemAlloc; freed once here.
            unsafe { CoTaskMemFree(Some(text.0.cast())) };
            name
        })
    })
    .and_then(|name| name)
    .map_err(|e| os(CaptureStage::Properties, &e))
}

/// The endpoint's id (IMMDevice::GetId).
fn endpoint_id(device: &IMMDevice) -> windows::core::Result<String> {
    // SAFETY: a live device; GetId returns a CoTaskMem string that take_string frees.
    take_string(unsafe { device.GetId() }?)
}

/// GetDefaultAudioEndpoint(eCapture, eConsole) → its id.
fn default_capture_id(enumerator: &IMMDeviceEnumerator) -> windows::core::Result<String> {
    // SAFETY: a live enumerator.
    let device = unsafe { enumerator.GetDefaultAudioEndpoint(eCapture, eConsole) }?;
    endpoint_id(&device)
}

/// A CoTaskMem wide string → String, freeing it. Invalid UTF-16 is
/// HRESULT_FROM_WIN32(ERROR_NO_UNICODE_TRANSLATION).
fn take_string(text: PWSTR) -> windows::core::Result<String> {
    // SAFETY: a valid, NUL-terminated CoTaskMem string from the caller's COM call.
    let decoded = unsafe { text.to_string() };
    // SAFETY: allocated by the COM callee with CoTaskMemAlloc; freed once here.
    unsafe { CoTaskMemFree(Some(text.0.cast())) };
    decoded.map_err(|_| WinError::from_hresult(ERROR_NO_UNICODE_TRANSLATION.to_hresult()))
}

/// OpenPropertyStore(STGM_READ) → GetValue(key) → parse → PropVariantClear.
fn read_property<T>(
    device: &IMMDevice,
    key: &PROPERTYKEY,
    parse: impl FnOnce(&PROPVARIANT) -> T,
) -> windows::core::Result<T> {
    // SAFETY: a live device; read access only.
    let store = unsafe { device.OpenPropertyStore(STGM_READ) }?;
    // SAFETY: a live store and a valid key.
    let mut value = unsafe { store.GetValue(key) }?;
    let parsed = parse(&value);
    // SAFETY: `value` came from GetValue and is cleared exactly once.
    unsafe { PropVariantClear(&mut value) }?;
    Ok(parsed)
}

/// A raw DEVICE_STATE → ours. Exactly one state bit is set for any endpoint.
fn device_state(raw: DEVICE_STATE) -> Result<DeviceState, CaptureError> {
    if raw == DEVICE_STATE_ACTIVE {
        Ok(DeviceState::Active)
    } else if raw == DEVICE_STATE_DISABLED {
        Ok(DeviceState::Disabled)
    } else if raw == DEVICE_STATE_NOTPRESENT {
        Ok(DeviceState::NotPresent)
    } else if raw == DEVICE_STATE_UNPLUGGED {
        Ok(DeviceState::Unplugged)
    } else {
        Err(CaptureError::Unexpected {
            stage: CaptureStage::Enumerate,
            detail: format!("illegal device state {}", raw.0),
        })
    }
}

fn os(stage: CaptureStage, e: &WinError) -> CaptureError {
    CaptureError::Os {
        stage,
        hresult: e.code().0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::E_OUTOFMEMORY;

    #[test]
    fn cp4_forward_guarded_counts_panics_and_failed_sends() {
        let dropped = AtomicU64::new(0);
        forward_guarded(&dropped, || panic!("cp4"));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);

        let dropped = AtomicU64::new(0);
        forward_guarded(&dropped, || {
            Err(TrySendError::Full(DeviceMsg::Added { id: "x".into() }))
        });
        assert_eq!(dropped.load(Ordering::Relaxed), 1);

        let dropped = AtomicU64::new(0);
        forward_guarded(&dropped, || Ok(()));
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn cp7_default_endpoint_id_maps_not_found_success_and_failure() {
        assert_eq!(
            default_endpoint_id(Err(WinError::from_hresult(HRESULT(0x8007_0490_u32 as i32)))),
            Ok(None)
        );
        assert_eq!(
            default_endpoint_id(Ok("{0.0.1.00000000}.{abc}".to_string())),
            Ok(Some("{0.0.1.00000000}.{abc}".to_string()))
        );
        assert_eq!(
            default_endpoint_id(Err(WinError::from_hresult(E_OUTOFMEMORY))),
            Err(CaptureError::Os {
                stage: CaptureStage::Enumerate,
                hresult: 0x8007_000E_u32 as i32,
            })
        );
    }

    /// Hardware: wasapi's notification registration succeeds and unregisters on drop.
    #[test]
    #[ignore = "hardware: registers device notifications with the system enumerator"]
    fn hardware_watch_registers_and_unregisters() {
        let _com = ComApartment::enter().expect("COM");
        let (tx, rx) = crossbeam_channel::bounded(64);
        let dropped = Arc::new(AtomicU64::new(0));
        let watch = watch(tx, Arc::clone(&dropped)).expect("registration");
        std::thread::sleep(std::time::Duration::from_millis(200));
        drop(watch);
        println!(
            "messages while watching: {:?}",
            rx.try_iter().collect::<Vec<_>>()
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
    }

    /// Hardware evidence (WP-C1): the full capture-device listing of this machine, one JSON line.
    /// Opens each Active endpoint briefly to probe it; no audio is read.
    #[test]
    #[ignore = "hardware: enumerates and probes this machine's capture endpoints"]
    fn hardware_list_capture_devices() {
        let devices = list_capture_devices().expect("listing");
        println!(
            "{}",
            serde_json::to_string_pretty(&devices).expect("DeviceInfo serializes")
        );
    }
}
