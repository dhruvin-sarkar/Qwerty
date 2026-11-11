//! Capture thread (p2 §2.3): one WASAPI shared-mode, event-driven capture stream (RAW when the
//! endpoint supports it and Initialize accepts it), moved packet by packet into the audio ring.
//!
//! The types §2.3 lists (`CaptureRequest`, `OpenedCapture`, `PacketMeta`, `CaptureExit`) live in
//! `messages.rs`, and the serde ones (`OpenedStream`, `MmcssStatus`, `CaptureStage`,
//! `CaptureError`, `DeviceProbe`) in `protocol.rs` (WP-C0); this module holds the behaviour.

use std::sync::Arc;
use std::thread::Thread;
use std::time::Duration;

use crossbeam_channel::{RecvTimeoutError, Sender, TrySendError};
use qwerty_core::audio::{AudioFormat, BlockTime, CaptureMode, DefaultModeReason, DeviceIdentity};
use qwerty_core::units::{ChannelCount, HostTime, Milliseconds, SampleIndex, SampleRate};
use wasapi::{
    AudioCaptureClient, AudioClient, AudioClientProperties, BufferInfo, Device, DeviceEnumerator,
    Direction, Handle, StreamCategory, StreamMode, StreamOption, WasapiError, WaveFormat,
};
use windows::core::{w, GUID, HRESULT};
use windows::Win32::Foundation::{
    CloseHandle, GetLastError, E_ACCESSDENIED, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::Media::Audio::{AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_E_DEVICE_IN_USE};
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, CreateEventW, SetEvent,
    WaitForMultipleObjects, WaitForSingleObject,
};

use crate::engine::devices::{self, from_wasapi_state};
use crate::engine::messages::{
    CaptureExit, CaptureRequest, OpenedCapture, PacketMeta, StopSignal, ThreadEnd, Worker,
};
use crate::engine::protocol::{
    CaptureError, CaptureStage, DeviceProbe, DeviceState, MmcssStatus, OpenedStream, ThreadName,
    CAPTURE_OPEN_TIMEOUT,
};
use crate::engine::ring::{audio_ring, PushOutcome, RingReader, RingWriter, CAPTURE_RING};

/// The event wait's timeout: 200 periods at 10 ms; no live device pauses that long.
pub const CAPTURE_STALL: Milliseconds = Milliseconds::new(2000.0);
/// CAPTURE_STALL for WaitForMultipleObjects.
const CAPTURE_STALL_MS: u32 = 2000;
/// The capture ring's packet slots (§2.3.1): 100 periods of headroom at 10 ms.
const CAPTURE_RING_PACKETS: usize = 1024;

/// mmreg.h WAVE_FORMAT_IEEE_FLOAT.
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
/// mmreg.h WAVE_FORMAT_EXTENSIBLE.
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// ksmedia.h KSDATAFORMAT_SUBTYPE_IEEE_FLOAT.
const SUBTYPE_IEEE_FLOAT: GUID = GUID::from_u128(0x0000_0003_0000_0010_8000_00aa_0038_9b71);

type OpenResult = Result<OpenedCapture, CaptureError>;

/// Called by WindowsParts::open_capture. Spawns the thread, which opens the stream and answers on a
/// bounded(1) channel; waits ≤ CAPTURE_OPEN_TIMEOUT (2 s) for it, else drops the Worker (signalling
/// stop) and returns OpenTimeout. Ok ⇒ the stream is started and the ring matches its final format.
pub fn open(request: CaptureRequest) -> Result<(Worker, OpenedCapture), CaptureError> {
    let CaptureRequest {
        worker,
        endpoint_id,
        wake,
        exits,
    } = request;
    let stop = Arc::new(StopEvent::new().map_err(|e| CaptureError::Os {
        stage: CaptureStage::Spawn,
        hresult: e.code().0,
    })?);
    let (opened_tx, opened_rx) = crossbeam_channel::bounded::<OpenResult>(1);
    let thread_stop = Arc::clone(&stop);
    let worker = Worker::spawn(
        worker,
        ThreadName::Capture,
        Box::new(CaptureStop(stop)),
        exits,
        move || ThreadEnd::Capture {
            exit: run(&endpoint_id, &wake, &thread_stop, &opened_tx),
        },
    )
    // SPEC: §2.3 names no CaptureError for a failed thread spawn; Worker::spawn's only error is
    // EngineError::ThreadSpawn, carried here as Unexpected at the Spawn stage.
    .map_err(|e| CaptureError::Unexpected {
        stage: CaptureStage::Spawn,
        detail: format!("{e:?}"),
    })?;
    let limit = Duration::from_secs_f64(CAPTURE_OPEN_TIMEOUT.get() / 1000.0);
    match opened_rx.recv_timeout(limit) {
        Ok(Ok(opened)) => Ok((worker, opened)),
        // The thread is already ending; dropping the Worker only signals stop.
        Ok(Err(error)) => Err(error),
        // Dropping the Worker signals stop: the thread releases the device without starting, or
        // stops the stream if it started meanwhile (§2.3 step 8).
        Err(RecvTimeoutError::Timeout) => Err(CaptureError::OpenTimeout),
        // SPEC: §2.3 does not say what open returns when the thread ends without answering (only
        // a panic in its body can do that; the controller also gets ThreadEnd::Panicked).
        Err(RecvTimeoutError::Disconnected) => Err(CaptureError::Unexpected {
            stage: CaptureStage::Spawn,
            detail: "the capture thread ended without an open result".to_string(),
        }),
    }
}

/// Probe without streaming: the mix format after the same set_properties capture uses, and RAW
/// support. For DeviceInfo. The caller thread is COM-initialized (MTA).
pub fn probe(endpoint_id: &str) -> Result<DeviceProbe, CaptureError> {
    let enumerator = DeviceEnumerator::new().map_err(|e| map_wasapi(CaptureStage::Enumerate, e))?;
    let device = capture_device(&enumerator, endpoint_id)?;
    let raw_supported = devices::raw_processing_supported(device.as_immdevice())?;
    let mix = mix_format(&device, raw_supported)?;
    Ok(DeviceProbe {
        format: mix.format,
        raw_supported,
    })
}

/// One exhaustive match over `WasapiError`, every variant named, no `_` arm (RUST-RT-25).
pub(crate) fn map_wasapi(stage: CaptureStage, e: WasapiError) -> CaptureError {
    match e {
        WasapiError::DeviceNotFound(endpoint_id) => CaptureError::DeviceNotFound { endpoint_id },
        WasapiError::Windows(e) => map_hresult(stage, e.code()),
        WasapiError::RegisterNotifications(e) => CaptureError::Os {
            stage,
            hresult: e.code().0,
        },
        // Impossible with our arguments (shared mode, capture direction, own format validation,
        // our own event wait); the pinned version makes a new variant a compile error.
        e @ (WasapiError::IllegalDeviceState(_)
        | WasapiError::IllegalDeviceRole(_)
        | WasapiError::IllegalDeviceDirection(_)
        | WasapiError::IllegalSessionState(_)
        | WasapiError::UnsupportedFormat
        | WasapiError::UnsupportedSubformat(_)
        | WasapiError::ClientNotInit
        | WasapiError::DataLengthMismatch { .. }
        | WasapiError::DataLengthTooShort { .. }
        | WasapiError::EventTimeout
        | WasapiError::AutomaticFormatConversionInExclusiveMode
        | WasapiError::LoopbackWithExclusiveMode
        | WasapiError::RenderToCaptureDevice
        | WasapiError::FromUtf16Error(_)) => CaptureError::Unexpected {
            stage,
            detail: e.to_string(),
        },
    }
}

/// A failing HRESULT → the named error, or Os{stage, hresult}.
fn map_hresult(stage: CaptureStage, hr: HRESULT) -> CaptureError {
    if hr == E_ACCESSDENIED {
        CaptureError::AccessDenied
    } else if hr == AUDCLNT_E_DEVICE_IN_USE {
        CaptureError::InUseExclusive
    } else if hr == AUDCLNT_E_DEVICE_INVALIDATED {
        CaptureError::DeviceInvalidated
    } else {
        CaptureError::Os {
            stage,
            hresult: hr.0,
        }
    }
}

/// A balanced `wasapi::initialize_mta()` / `deinitialize()` on this thread. S_FALSE (already MTA)
/// counts as entered, and is balanced too.
pub(crate) struct ComApartment {
    // Not Send: COM initialization belongs to the thread that entered it.
    _thread_bound: std::marker::PhantomData<*const ()>,
}

impl ComApartment {
    pub(crate) fn enter() -> Result<ComApartment, CaptureError> {
        let hr = wasapi::initialize_mta();
        if hr.is_err() {
            return Err(CaptureError::ComInit { hresult: hr.0 });
        }
        Ok(ComApartment {
            _thread_bound: std::marker::PhantomData,
        })
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        wasapi::deinitialize();
    }
}

/// The capture thread's manual-reset stop event; the second handle of its wait.
struct StopEvent {
    handle: HANDLE,
}

// SAFETY: a Win32 event handle is a process-wide kernel object handle; SetEvent and the waits are
// thread-safe, and the handle is closed only in Drop, after every Arc holder is gone.
unsafe impl Send for StopEvent {}
// SAFETY: as above; &StopEvent only exposes SetEvent, a zero-timeout wait and the raw handle.
unsafe impl Sync for StopEvent {}

impl StopEvent {
    fn new() -> windows::core::Result<StopEvent> {
        // SAFETY: no security attributes and no name; manual reset, initially unset; the returned
        // handle is owned by StopEvent.
        let handle = unsafe { CreateEventW(None, true, false, None) }?;
        Ok(StopEvent { handle })
    }

    fn set(&self) {
        // SAFETY: `handle` is a valid event handle until Drop.
        let set = unsafe { SetEvent(self.handle) };
        // INVARIANT: SetEvent fails only on an invalid handle, and this one stays open until Drop.
        set.expect("SetEvent on an open event handle cannot fail");
    }

    fn is_set(&self) -> bool {
        // SAFETY: `handle` is a valid event handle until Drop; a zero timeout never blocks.
        let woke = unsafe { WaitForSingleObject(self.handle, 0) };
        woke == WAIT_OBJECT_0
    }
}

impl Drop for StopEvent {
    fn drop(&mut self) {
        // SAFETY: the handle was created in `new` and is closed only here.
        if let Err(e) = unsafe { CloseHandle(self.handle) } {
            tracing::error!(
                hresult = e.code().0,
                "CloseHandle of the capture stop event failed"
            );
        }
    }
}

/// The Worker's stop signal: SetEvent on the stop event (idempotent, non-blocking).
struct CaptureStop(Arc<StopEvent>);

impl StopSignal for CaptureStop {
    fn signal(&self) {
        self.0.set();
    }
}

/// The thread body: COM for the thread's lifetime around one capture session.
fn run(
    endpoint_id: &str,
    wake: &Thread,
    stop: &StopEvent,
    opened: &Sender<OpenResult>,
) -> CaptureExit {
    let _com = match ComApartment::enter() {
        Ok(com) => com,
        Err(error) => return fail(opened, error),
    };
    // Every COM object lives inside `session` and is released before `_com` uninitializes.
    session(endpoint_id, wake, stop, opened)
}

/// Open (steps 2-8), stream (the loop), and release.
fn session(
    endpoint_id: &str,
    wake: &Thread,
    stop: &StopEvent,
    opened: &Sender<OpenResult>,
) -> CaptureExit {
    let (mut stream, facts, reader) = match open_stream(endpoint_id) {
        Ok(parts) => parts,
        Err(error) => return fail(opened, error),
    };
    // Step 7. Declared after `stream`, so it reverts before the COM objects are released.
    let mmcss = Mmcss::enter();
    // Built before start_stream: nothing after it allocates.
    let answer = OpenedCapture {
        stream: OpenedStream {
            identity: facts.identity,
            period: facts.period,
            buffer_frames: facts.buffer_frames,
            mmcss: mmcss.status,
        },
        ring: reader,
    };
    // Step 8.
    if stop.is_set() {
        return gave_up();
    }
    if let Err(e) = stream.client.start_stream() {
        return fail(opened, map_wasapi(CaptureStage::Start, e));
    }
    match opened.try_send(Ok(answer)) {
        Ok(()) => {}
        // The only send on a bounded(1) channel is never Full; Disconnected means open() gave up
        // (RUST-RT-7).
        Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
            stop_stream(&stream.client);
            return gave_up();
        }
    }
    let error = stream.pump(wake, stop);
    stop_stream(&stream.client);
    stream.sink.exit(error)
}

/// An open failure: answer open() (unless it already gave up) and end with the error.
fn fail(opened: &Sender<OpenResult>, error: CaptureError) -> CaptureExit {
    match opened.try_send(Err(error.clone())) {
        // Never Full (the only send); Disconnected means open() already returned OpenTimeout,
        // and the real error is reported in this exit.
        Ok(()) | Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {}
    }
    CaptureExit {
        frames: 0,
        dropped_frames: 0,
        packets: 0,
        error: Some(error),
    }
}

/// The controller gave up waiting (step 8): end without streaming.
fn gave_up() -> CaptureExit {
    CaptureExit {
        frames: 0,
        dropped_frames: 0,
        packets: 0,
        error: Some(CaptureError::OpenTimeout),
    }
}

fn stop_stream(client: &AudioClient) {
    if let Err(e) = client.stop_stream() {
        // After an invalidation the stream cannot be stopped; the exit already names the cause.
        tracing::debug!(error = ?map_wasapi(CaptureStage::Read, e), "stop_stream failed");
    }
}

/// MMCSS "Audio" for the capture thread (step 7): not fatal, reverted on drop.
struct Mmcss {
    handle: Option<HANDLE>,
    status: MmcssStatus,
}

impl Mmcss {
    fn enter() -> Mmcss {
        let mut task_index = 0u32;
        // SAFETY: a static task name and a valid out pointer.
        match unsafe { AvSetMmThreadCharacteristicsW(w!("Audio"), &mut task_index) } {
            Ok(handle) => Mmcss {
                handle: Some(handle),
                status: MmcssStatus::Active,
            },
            Err(e) => Mmcss {
                handle: None,
                status: MmcssStatus::Unavailable {
                    code: win32_code(e.code()),
                },
            },
        }
    }
}

impl Drop for Mmcss {
    fn drop(&mut self) {
        if let Some(handle) = self.handle {
            // SAFETY: the handle AvSetMmThreadCharacteristicsW returned on this thread.
            if let Err(e) = unsafe { AvRevertMmThreadCharacteristics(handle) } {
                tracing::debug!(
                    hresult = e.code().0,
                    "AvRevertMmThreadCharacteristics failed"
                );
            }
        }
    }
}

/// GetLastError's code from the HRESULT windows-rs built from it (HRESULT_FROM_WIN32).
fn win32_code(hr: HRESULT) -> u32 {
    let bits = hr.0 as u32;
    if bits & 0xFFFF_0000 == 0x8007_0000 {
        bits & 0xFFFF
    } else {
        bits
    }
}

/// Step 2: the endpoint, which must be a capture endpoint and Active.
fn capture_device(
    enumerator: &DeviceEnumerator,
    endpoint_id: &str,
) -> Result<Device, CaptureError> {
    let not_found = || CaptureError::DeviceNotFound {
        endpoint_id: endpoint_id.to_string(),
    };
    let device = enumerator
        .get_device(endpoint_id)
        .map_err(|_| not_found())?;
    match device.get_direction() {
        Direction::Capture => {}
        // wasapi would open a render endpoint as a loopback capture of the speaker mix.
        Direction::Render => return Err(not_found()),
    }
    let state = device
        .get_state()
        .map_err(|e| map_wasapi(CaptureStage::Enumerate, e))?;
    match from_wasapi_state(state) {
        DeviceState::Active => Ok(device),
        state @ (DeviceState::Disabled | DeviceState::NotPresent | DeviceState::Unplugged) => {
            Err(CaptureError::DeviceNotActive {
                endpoint_id: endpoint_id.to_string(),
                state,
            })
        }
    }
}

/// An activated client after set_properties (category Other, plus RAW when `raw`), with its
/// validated mix format.
struct MixFormat {
    client: AudioClient,
    wave: WaveFormat,
    format: AudioFormat,
}

fn mix_format(device: &Device, raw: bool) -> Result<MixFormat, CaptureError> {
    let client = device
        .get_iaudioclient()
        .map_err(|e| map_wasapi(CaptureStage::Activate, e))?;
    let properties = AudioClientProperties::new().set_category(StreamCategory::Other);
    let properties = if raw {
        properties.set_option(StreamOption::Raw)
    } else {
        properties
    };
    client
        .set_properties(properties)
        .map_err(|e| map_wasapi(CaptureStage::Properties, e))?;
    let wave = client
        .get_mixformat()
        .map_err(|e| map_wasapi(CaptureStage::MixFormat, e))?;
    let format = validate_mix_format(&wave)?;
    Ok(MixFormat {
        client,
        wave,
        format,
    })
}

/// Step 5a on a wasapi WaveFormat.
fn validate_mix_format(wave: &WaveFormat) -> Result<AudioFormat, CaptureError> {
    // Copies out of the packed struct.
    let tag = wave.wave_fmt.Format.wFormatTag;
    let subformat = wave.wave_fmt.SubFormat;
    validate_format(
        tag,
        subformat,
        wave.get_bitspersample(),
        wave.get_samplespersec(),
        wave.get_nchannels(),
    )
}

/// 32-bit IEEE float (tag 3, or EXTENSIBLE + float subformat), rate 16-192 kHz, 1-32 channels.
/// Otherwise the named `Unsupported*` error. Never autoconvert.
fn validate_format(
    tag: u16,
    subformat: GUID,
    bits: u16,
    hz: u32,
    channels: u16,
) -> Result<AudioFormat, CaptureError> {
    let float = match tag {
        WAVE_FORMAT_IEEE_FLOAT => true,
        WAVE_FORMAT_EXTENSIBLE => subformat == SUBTYPE_IEEE_FLOAT,
        _ => false,
    };
    if !float || bits != 32 {
        // A KSDATAFORMAT subtype's Data1 is the format tag it stands for, so it fits u16.
        let tag = if tag == WAVE_FORMAT_EXTENSIBLE {
            subformat.data1 as u16
        } else {
            tag
        };
        return Err(CaptureError::UnsupportedMixFormat { tag, bits });
    }
    let rate = SampleRate::new(hz).map_err(|_| CaptureError::UnsupportedSampleRate { hz })?;
    let channels =
        ChannelCount::new(channels).map_err(|_| CaptureError::UnsupportedChannelCount {
            channels: u32::from(channels),
        })?;
    Ok(AudioFormat { rate, channels })
}

/// The open stream and everything the loop touches, all allocated before start_stream.
struct Stream {
    // Field order is drop order: the capture client, then the client, then its event handle.
    capture: AudioCaptureClient,
    client: AudioClient,
    event: Handle,
    block_align: usize,
    ring: RingWriter,
    bytes: Vec<u8>,
    sink: PacketSink,
}

/// What open() reports about the stream (OpenedStream without the MMCSS status).
struct StreamFacts {
    identity: DeviceIdentity,
    period: Milliseconds,
    buffer_frames: u32,
}

/// Steps 2-6.
fn open_stream(endpoint_id: &str) -> Result<(Stream, StreamFacts, RingReader), CaptureError> {
    let enumerator = DeviceEnumerator::new().map_err(|e| map_wasapi(CaptureStage::Enumerate, e))?;
    let device = capture_device(&enumerator, endpoint_id)?;
    // Step 3.
    let raw_supported = devices::raw_processing_supported(device.as_immdevice())?;
    let name = devices::friendly_name(device.as_immdevice())?;
    // Steps 4-5.
    let (mix, capture_mode, default_period) = if raw_supported {
        let mut mix = mix_format(&device, true)?;
        let default_period = device_period(&mix.client)?;
        match mix.client.initialize_client(
            &mix.wave,
            &Direction::Capture,
            &events_shared(default_period),
        ) {
            Ok(()) => (mix, CaptureMode::Raw, default_period),
            Err(WasapiError::Windows(e)) => {
                drop(mix);
                initialize_default(
                    &device,
                    DefaultModeReason::RawRejected {
                        hresult: e.code().0,
                    },
                )?
            }
            Err(e) => return Err(map_wasapi(CaptureStage::Initialize, e)),
        }
    } else {
        initialize_default(&device, DefaultModeReason::RawNotSupported)?
    };
    // Step 6.
    let event = mix
        .client
        .set_get_eventhandle()
        .map_err(|e| map_wasapi(CaptureStage::EventHandle, e))?;
    let capture = mix
        .client
        .get_audiocaptureclient()
        .map_err(|e| map_wasapi(CaptureStage::CaptureClient, e))?;
    // SPEC: §2.3 names no stage for GetBufferSize or GetDevicePeriod; both are Initialize.
    let buffer_frames = mix
        .client
        .get_buffer_size()
        .map_err(|e| map_wasapi(CaptureStage::Initialize, e))?;
    // After the final GetMixFormat, so the ring's channel count is always the stream's (RUST-RT-11).
    let (ring, reader) = audio_ring(
        mix.format,
        CAPTURE_RING,
        CAPTURE_RING_PACKETS,
        buffer_frames,
    );
    let block_align = mix.wave.get_blockalign() as usize;
    // The last allocations before the stream starts.
    let bytes = vec![0u8; buffer_frames as usize * block_align];
    let sink = PacketSink::new(mix.format.channels, buffer_frames);
    let identity = DeviceIdentity {
        endpoint_id: endpoint_id.to_string(),
        name,
        format: mix.format,
        capture_mode,
    };
    let stream = Stream {
        capture,
        client: mix.client,
        event,
        block_align,
        ring,
        bytes,
        sink,
    };
    let facts = StreamFacts {
        identity,
        period: milliseconds_from_hns(default_period),
        buffer_frames,
    };
    Ok((stream, facts, reader))
}

/// Step 5: the Default path (no RAW option).
fn initialize_default(
    device: &Device,
    reason: DefaultModeReason,
) -> Result<(MixFormat, CaptureMode, i64), CaptureError> {
    let mut mix = mix_format(device, false)?;
    let default_period = device_period(&mix.client)?;
    mix.client
        .initialize_client(
            &mix.wave,
            &Direction::Capture,
            &events_shared(default_period),
        )
        .map_err(|e| map_wasapi(CaptureStage::Initialize, e))?;
    Ok((mix, CaptureMode::Default { reason }, default_period))
}

/// The engine's default period, in 100-ns units.
fn device_period(client: &AudioClient) -> Result<i64, CaptureError> {
    let (default_period, _minimum) = client
        .get_device_period()
        .map_err(|e| map_wasapi(CaptureStage::Initialize, e))?;
    Ok(default_period)
}

/// Shared, event-driven, no autoconvert, a buffer of two default periods.
fn events_shared(default_period: i64) -> StreamMode {
    StreamMode::EventsShared {
        autoconvert: false,
        buffer_duration_hns: 2 * default_period,
    }
}

fn milliseconds_from_hns(hns: i64) -> Milliseconds {
    // INVARIANT: an i64 divided by 10^4 is a finite f64.
    Milliseconds::try_new(hns as f64 / 10_000.0).expect("a 100-ns count is a finite duration")
}

impl Stream {
    /// The loop. Between start_stream and stop_stream: no allocation, no lock, no channel, no
    /// tracing (RUST-RT-23, RUST-RT-31). Returns why it ended (None: stop).
    fn pump(&mut self, wake: &Thread, stop: &StopEvent) -> Option<CaptureError> {
        let handles = [self.event.as_handle(), stop.handle];
        loop {
            // SAFETY: both handles stay open for the whole loop (owned by self and the caller).
            let woke = unsafe { WaitForMultipleObjects(&handles, false, CAPTURE_STALL_MS) };
            if woke == WAIT_OBJECT_0 {
                if let Err(error) = self.drain(wake) {
                    return Some(error);
                }
            } else if woke.0 == WAIT_OBJECT_0.0 + 1 {
                return None;
            } else if woke == WAIT_TIMEOUT {
                return Some(CaptureError::Stalled {
                    after: CAPTURE_STALL,
                });
            } else {
                // WAIT_FAILED: events cannot be abandoned.
                // SAFETY: reads this thread's last-error value.
                let code = unsafe { GetLastError() };
                return Some(CaptureError::Os {
                    stage: CaptureStage::Read,
                    hresult: code.to_hresult().0,
                });
            }
        }
    }

    /// Reads every pending packet into the ring, waking processing after each.
    fn drain(&mut self, wake: &Thread) -> Result<(), CaptureError> {
        loop {
            match self
                .capture
                .get_next_packet_size()
                .map_err(|e| map_wasapi(CaptureStage::Read, e))?
            {
                Some(0) | None => return Ok(()),
                Some(_) => {
                    let (frames, info) = self
                        .capture
                        .read_from_device(&mut self.bytes)
                        .map_err(|e| map_wasapi(CaptureStage::Read, e))?;
                    if frames == 0 {
                        return Ok(());
                    }
                    let packet = &self.bytes[..frames as usize * self.block_align];
                    self.sink.deliver(&mut self.ring, packet, frames, &info);
                    wake.unpark();
                }
            }
        }
    }
}

/// Little-endian f32 samples, or zeros if the packet is flagged silent.
pub(crate) fn convert_packet(bytes: &[u8], silent: bool, out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len() * 4, "a packet is 4 bytes per sample");
    if silent {
        out.fill(0.0);
        return;
    }
    for (sample, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *sample = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
}

/// Device positions → our strictly increasing `SampleIndex` (RUST-RT-12).
///
/// `start` is the device position minus a base set by the first packet, so every lost frame (a
/// device-side gap or one of our ring drops) shows as a jump in `start`. Tracking the next
/// expected device position is the same as tracking the base: `expected = base + next_start`.
#[derive(Debug, Default)]
pub(crate) struct PositionTracker {
    /// The device position the next packet should start at; None before the first packet.
    expected: Option<u64>,
    /// Our `start` for that next packet.
    next_start: u64,
    /// A packet was dropped at the ring and no packet has been written since.
    ring_drop_pending: bool,
}

impl PositionTracker {
    /// (start, discontinuity, frames lost on the device side).
    pub(crate) fn next(
        &mut self,
        index: u64,
        frames: u32,
        data_discontinuity: bool,
    ) -> (SampleIndex, bool, u64) {
        let expected = *self.expected.get_or_insert(index);
        let (start, lost, jumped) = if index > expected {
            // Forward: the gap is lost and shows in `start`.
            let lost = index - expected;
            (self.next_start + lost, lost, true)
        } else if index < expected {
            // Backwards: re-base, so `start` stays strictly increasing; nothing is lost.
            (self.next_start, 0, true)
        } else {
            (self.next_start, 0, false)
        };
        self.expected = Some(index + u64::from(frames));
        self.next_start = start + u64::from(frames);
        let discontinuity = jumped || data_discontinuity || self.ring_drop_pending;
        (SampleIndex(start), discontinuity, lost)
    }

    /// The packet `next` described was written to the ring.
    pub(crate) fn written(&mut self) {
        self.ring_drop_pending = false;
    }

    /// The packet `next` described was dropped at the ring: the next written one carries
    /// `discontinuity`.
    pub(crate) fn ring_dropped(&mut self) {
        self.ring_drop_pending = true;
    }
}

/// The per-packet steps 1-4 and the exit counters.
pub(crate) struct PacketSink {
    positions: PositionTracker,
    /// buffer_frames × channels, preallocated.
    samples: Vec<f32>,
    channels: usize,
    frames: u64,
    dropped_frames: u64,
    packets: u64,
}

impl PacketSink {
    pub(crate) fn new(channels: ChannelCount, max_frames: u32) -> PacketSink {
        let channels = usize::from(channels.get());
        PacketSink {
            positions: PositionTracker::default(),
            samples: vec![0.0; max_frames as usize * channels],
            channels,
            frames: 0,
            dropped_frames: 0,
            packets: 0,
        }
    }

    /// Convert, position, time and push one packet of `frames` frames (`bytes` holds exactly
    /// them; frames ≤ max_frames because WASAPI never returns more than the buffer holds).
    pub(crate) fn deliver(
        &mut self,
        ring: &mut RingWriter,
        bytes: &[u8],
        frames: u32,
        info: &BufferInfo,
    ) -> PushOutcome {
        let samples = &mut self.samples[..frames as usize * self.channels];
        convert_packet(bytes, info.flags.silent, samples);
        let (start, discontinuity, lost) =
            self.positions
                .next(info.index, frames, info.flags.data_discontinuity);
        let time = if info.flags.timestamp_error {
            BlockTime::Untimed
        } else {
            // QPC in 100 ns, the HostTime domain; far below i64::MAX.
            BlockTime::Timestamped {
                host_time: HostTime(info.timestamp as i64),
            }
        };
        self.packets += 1;
        self.frames += u64::from(frames);
        self.dropped_frames += lost;
        let meta = PacketMeta {
            start,
            frames,
            time,
            discontinuity,
        };
        let outcome = ring.push(meta, samples);
        match outcome {
            PushOutcome::Written => self.positions.written(),
            PushOutcome::Dropped { frames } => {
                self.dropped_frames += u64::from(frames);
                self.positions.ring_dropped();
            }
        }
        outcome
    }

    pub(crate) fn exit(&self, error: Option<CaptureError>) -> CaptureExit {
        CaptureExit {
            frames: self.frames,
            dropped_frames: self.dropped_frames,
            packets: self.packets,
            error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::Error as WinError;
    use windows::Win32::Media::Audio::AUDCLNT_BUFFERFLAGS_SILENT;

    fn mono_48k() -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(48_000).expect("rate"),
            channels: ChannelCount::new(1).expect("channels"),
        }
    }

    /// `frames` mono frames whose sample values are their device positions.
    fn packet_bytes(index: u64, frames: u32) -> Vec<u8> {
        (0..u64::from(frames))
            .flat_map(|i| ((index + i) as f32).to_le_bytes())
            .collect()
    }

    fn info(index: u64, flags: u32) -> BufferInfo {
        BufferInfo::new(flags, index, 1_000 + index)
    }

    #[test]
    fn cp1_silent_flag_gives_zeros() {
        let bytes = packet_bytes(7, 4);
        let mut out = [9.0f32; 4];
        convert_packet(&bytes, true, &mut out);
        assert_eq!(out, [0.0; 4]);
        convert_packet(&bytes, false, &mut out);
        assert_eq!(out, [7.0, 8.0, 9.0, 10.0]);

        // Through the sink: the silent packet reaches the ring as zeros.
        let (mut ring, mut reader) = audio_ring(mono_48k(), CAPTURE_RING, 16, 4);
        let mut sink = PacketSink::new(mono_48k().channels, 4);
        let flags = AUDCLNT_BUFFERFLAGS_SILENT.0 as u32;
        sink.deliver(&mut ring, &bytes, 4, &info(7, flags));
        let mut popped = Vec::new();
        reader.pop(&mut popped).expect("a packet");
        assert_eq!(popped, vec![0.0; 4]);
    }

    #[test]
    fn cp2_forward_gap_is_a_counted_discontinuity() {
        let (mut ring, mut reader) = audio_ring(mono_48k(), CAPTURE_RING, 16, 10);
        let mut sink = PacketSink::new(mono_48k().channels, 10);
        sink.deliver(&mut ring, &packet_bytes(100, 10), 10, &info(100, 0));
        // Device positions 110..125 never arrived.
        sink.deliver(&mut ring, &packet_bytes(125, 10), 10, &info(125, 0));
        let mut out = Vec::new();
        let first = reader.pop(&mut out).expect("first");
        assert_eq!((first.start, first.discontinuity), (SampleIndex(0), false));
        let second = reader.pop(&mut out).expect("second");
        assert_eq!(second.start, SampleIndex(25));
        assert!(second.discontinuity);
        assert_eq!(
            second.time,
            BlockTime::Timestamped {
                host_time: HostTime(1_125)
            }
        );
        let exit = sink.exit(None);
        assert_eq!(
            (exit.frames, exit.dropped_frames, exit.packets),
            (20, 15, 2)
        );
    }

    #[test]
    fn cp3_ring_full_marks_the_next_written_packet() {
        // Room for exactly one 10-frame packet.
        let (mut ring, mut reader) = audio_ring(mono_48k(), Milliseconds::new(0.0), 4, 10);
        let mut sink = PacketSink::new(mono_48k().channels, 10);
        assert_eq!(
            sink.deliver(&mut ring, &packet_bytes(0, 10), 10, &info(0, 0)),
            PushOutcome::Written
        );
        assert_eq!(
            sink.deliver(&mut ring, &packet_bytes(10, 10), 10, &info(10, 0)),
            PushOutcome::Dropped { frames: 10 }
        );
        let mut out = Vec::new();
        let first = reader.pop(&mut out).expect("first");
        assert_eq!((first.start, first.discontinuity), (SampleIndex(0), false));
        assert_eq!(
            sink.deliver(&mut ring, &packet_bytes(20, 10), 10, &info(20, 0)),
            PushOutcome::Written
        );
        let third = reader.pop(&mut out).expect("third");
        assert_eq!(
            third.start,
            SampleIndex(20),
            "the dropped packet shows as a gap"
        );
        assert!(third.discontinuity);
        assert_eq!(out, (20..30).map(|v| v as f32).collect::<Vec<_>>());
        assert_eq!(reader.pop(&mut out), None);
        let exit = sink.exit(None);
        assert_eq!(
            (exit.frames, exit.dropped_frames, exit.packets),
            (30, 10, 3)
        );
        // The flag is spent once a packet is written.
        assert_eq!(
            sink.deliver(&mut ring, &packet_bytes(30, 10), 10, &info(30, 0)),
            PushOutcome::Written
        );
        assert!(!reader.pop(&mut out).expect("fourth").discontinuity);
    }

    #[test]
    fn cp5_backwards_jump_rebases() {
        let mut positions = PositionTracker::default();
        assert_eq!(positions.next(1_000, 10, false), (SampleIndex(0), false, 0));
        positions.written();
        assert_eq!(
            positions.next(1_010, 10, false),
            (SampleIndex(10), false, 0)
        );
        positions.written();
        // The device position moves back to 0 (below our next start of 20, too).
        assert_eq!(positions.next(0, 10, false), (SampleIndex(20), true, 0));
        positions.written();
        assert_eq!(positions.next(10, 10, false), (SampleIndex(30), false, 0));
        positions.written();
        // data_discontinuity alone also marks the packet.
        assert_eq!(positions.next(20, 10, true), (SampleIndex(40), true, 0));
    }

    #[test]
    fn cp6_map_wasapi_rows() {
        let stage = CaptureStage::Initialize;
        let win = |hr: HRESULT| WasapiError::Windows(WinError::from_hresult(hr));
        assert_eq!(
            map_wasapi(stage, WasapiError::DeviceNotFound("id".into())),
            CaptureError::DeviceNotFound {
                endpoint_id: "id".into()
            }
        );
        assert_eq!(
            map_wasapi(stage, win(E_ACCESSDENIED)),
            CaptureError::AccessDenied
        );
        assert_eq!(
            map_wasapi(stage, win(AUDCLNT_E_DEVICE_IN_USE)),
            CaptureError::InUseExclusive
        );
        assert_eq!(
            map_wasapi(stage, win(AUDCLNT_E_DEVICE_INVALIDATED)),
            CaptureError::DeviceInvalidated
        );
        // AUDCLNT_E_UNSUPPORTED_FORMAT: any other HRESULT keeps its stage.
        let other = HRESULT(0x8889_0008_u32 as i32);
        assert_eq!(
            map_wasapi(stage, win(other)),
            CaptureError::Os {
                stage,
                hresult: other.0
            }
        );
        assert_eq!(
            map_wasapi(
                CaptureStage::Enumerate,
                WasapiError::RegisterNotifications(WinError::from_hresult(E_ACCESSDENIED))
            ),
            CaptureError::Os {
                stage: CaptureStage::Enumerate,
                hresult: E_ACCESSDENIED.0
            }
        );
        let utf16 = String::from_utf16(&[0xD800]).expect_err("an unpaired surrogate");
        let impossible = [
            WasapiError::IllegalDeviceState(99),
            WasapiError::IllegalDeviceRole(9),
            WasapiError::IllegalDeviceDirection(9),
            WasapiError::IllegalSessionState(9),
            WasapiError::UnsupportedFormat,
            WasapiError::UnsupportedSubformat(GUID::zeroed()),
            WasapiError::ClientNotInit,
            WasapiError::DataLengthMismatch {
                received: 1,
                expected: 2,
            },
            WasapiError::DataLengthTooShort {
                received: 1,
                expected: 2,
            },
            WasapiError::EventTimeout,
            WasapiError::AutomaticFormatConversionInExclusiveMode,
            WasapiError::LoopbackWithExclusiveMode,
            WasapiError::RenderToCaptureDevice,
            WasapiError::FromUtf16Error(utf16),
        ];
        for e in impossible {
            let detail = e.to_string();
            assert_eq!(
                map_wasapi(CaptureStage::Read, e),
                CaptureError::Unexpected {
                    stage: CaptureStage::Read,
                    detail
                }
            );
        }
    }

    #[test]
    fn validate_format_accepts_only_32_bit_float_in_range() {
        let none = GUID::zeroed();
        let pcm = GUID::from_u128(0x0000_0001_0000_0010_8000_00aa_0038_9b71);
        let ok = Ok(AudioFormat {
            rate: SampleRate::new(48_000).expect("rate"),
            channels: ChannelCount::new(2).expect("channels"),
        });
        assert_eq!(validate_format(3, none, 32, 48_000, 2), ok);
        assert_eq!(
            validate_format(WAVE_FORMAT_EXTENSIBLE, SUBTYPE_IEEE_FLOAT, 32, 48_000, 2),
            ok
        );
        assert_eq!(
            validate_format(WAVE_FORMAT_EXTENSIBLE, pcm, 16, 48_000, 2),
            Err(CaptureError::UnsupportedMixFormat { tag: 1, bits: 16 })
        );
        assert_eq!(
            validate_format(3, none, 64, 48_000, 2),
            Err(CaptureError::UnsupportedMixFormat { tag: 3, bits: 64 })
        );
        assert_eq!(
            validate_format(3, none, 32, 8_000, 2),
            Err(CaptureError::UnsupportedSampleRate { hz: 8_000 })
        );
        assert_eq!(
            validate_format(3, none, 32, 48_000, 33),
            Err(CaptureError::UnsupportedChannelCount { channels: 33 })
        );
    }

    /// One mix format, as GetMixFormat reported it, for the OQ-1 evidence.
    fn describe(wave: &WaveFormat) -> serde_json::Value {
        let tag = wave.wave_fmt.Format.wFormatTag;
        let subformat = wave.wave_fmt.SubFormat;
        serde_json::json!({
            "rate": wave.get_samplespersec(),
            "channels": wave.get_nchannels(),
            "bits": wave.get_bitspersample(),
            "tag": format!("{tag:#06x}"),
            "subformat_data1": format!("{:#010x}", subformat.data1),
            "channel_mask": format!("{:#x}", wave.get_dwchannelmask()),
            "validated": format!("{:?}", validate_mix_format(wave)),
        })
    }

    /// GetMixFormat after set_properties (RAW or Default), then Initialize as capture would.
    fn measure(device: &Device, raw: bool) -> serde_json::Value {
        let mut mix = match mix_format(device, raw) {
            Ok(mix) => mix,
            Err(error) => return serde_json::json!({ "error": format!("{error:?}") }),
        };
        let initialize = device_period(&mix.client).and_then(|period| {
            mix.client
                .initialize_client(&mix.wave, &Direction::Capture, &events_shared(period))
                .map_err(|e| map_wasapi(CaptureStage::Initialize, e))
                .map(|()| period)
        });
        serde_json::json!({
            "mix_format": describe(&mix.wave),
            "initialize": format!("{initialize:?}"),
        })
    }

    /// Hardware evidence (OQ-1): for every Active capture endpoint, GetMixFormat after RAW
    /// set_properties vs after Default set_properties, and whether Initialize accepts each. Opens
    /// each endpoint briefly; never starts a stream and reads no audio.
    #[test]
    #[ignore = "hardware: opens every active capture endpoint briefly (OQ-1)"]
    fn hardware_oq1_raw_vs_default_mix_format() {
        let _com = ComApartment::enter().expect("COM");
        let enumerator = DeviceEnumerator::new().expect("enumerator");
        let active = enumerator
            .get_device_collection(&Direction::Capture)
            .expect("active capture endpoints");
        let count = active.get_nbr_devices().expect("count");
        for index in 0..count {
            let device = active.get_device_at_index(index).expect("device");
            let raw_supported = devices::raw_processing_supported(device.as_immdevice());
            let line = serde_json::json!({
                "endpoint_id": device.get_id().expect("id"),
                "name": devices::friendly_name(device.as_immdevice()).expect("name"),
                "raw_supported": format!("{raw_supported:?}"),
                "raw": measure(&device, true),
                "default": measure(&device, false),
            });
            println!("{line}");
        }
    }

    /// Hardware smoke test: opens the default microphone through `open`, drains the ring for 5 s,
    /// stops, and prints the stream facts and counters. No audio is kept.
    #[test]
    #[ignore = "hardware: captures 5 s from the default microphone; no audio is kept"]
    fn hardware_capture_default_microphone() {
        let devices = devices::list_capture_devices().expect("listing");
        let default = devices
            .iter()
            .find(|d| d.is_default)
            .expect("a default capture endpoint");
        let (exits_tx, exits_rx) = crossbeam_channel::bounded(16);
        let (worker, mut opened) = open(CaptureRequest {
            worker: crate::engine::messages::WorkerId(1),
            endpoint_id: default.endpoint_id.clone(),
            wake: std::thread::current(),
            exits: exits_tx,
        })
        .expect("open");
        let mut samples = Vec::new();
        let (mut packets, mut frames, mut discontinuities) = (0u64, 0u64, 0u64);
        let mut last: Option<PacketMeta> = None;
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_secs(5) {
            std::thread::park_timeout(Duration::from_millis(50));
            while let Some(meta) = opened.ring.pop(&mut samples) {
                if let Some(previous) = last {
                    assert!(meta.start.0 >= previous.start.0 + u64::from(previous.frames));
                }
                assert!(samples.iter().all(|s| s.is_finite()));
                packets += 1;
                frames += u64::from(meta.frames);
                discontinuities += u64::from(meta.discontinuity);
                last = Some(meta);
            }
        }
        worker.stop(Duration::from_secs(2)).expect("joined");
        let exit = exits_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("thread exit");
        let ThreadEnd::Capture { exit } = exit.end else {
            panic!("the capture thread ended with {:?}", exit.end);
        };
        let line = serde_json::json!({
            "stream": opened.stream,
            "popped": { "packets": packets, "frames": frames, "discontinuities": discontinuities },
            "exit": exit,
        });
        println!("{line}");
        assert_eq!(exit.error, None);
        assert!(packets > 0);
    }
}
