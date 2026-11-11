//! `WindowsParts`, the product `OsParts` (p2 §2.2): it wires capture and devices (WP-C1), raw input
//! (WP-C3) and the platform (WP-C2) to the engine's one OS seam. `testkit::FakeOs` is the other
//! implementation.

use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Sender;

use qwerty_core::units::HostTime;
use windows::core::PCWSTR;
use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

use crate::engine::clock;
use crate::engine::messages::{
    CaptureRequest, DeviceMsg, DeviceWatch, OpenedCapture, OsParts, RawInputRequest, RecorderDisk,
    Worker,
};
use crate::engine::protocol::{CaptureError, CaptureStage, DeviceInfo, VetoUnavailableReason};
use crate::platform::{OutputObserver, PlatformActions};

/// The product OS parts: QPC time, WASAPI capture and devices, Raw Input, the Windows platform.
#[derive(Debug, Default)]
pub struct WindowsParts;

impl WindowsParts {
    pub fn new() -> WindowsParts {
        WindowsParts
    }
}

/// The text of a part that WP-C1/C3 provide and the integration step wires.
const NOT_WIRED: &str = "this part is wired when WP-C1..C3 land";

impl OsParts for WindowsParts {
    fn now(&self) -> HostTime {
        clock::now()
    }

    /// deadline − now, or zero once it has passed.
    fn timeout_until(&self, deadline: HostTime) -> Duration {
        let left = deadline.since(clock::now()).0;
        if left <= 0 {
            Duration::ZERO
        } else {
            // 100-ns ticks.
            Duration::from_nanos(left as u64 * 100)
        }
    }

    fn list_devices(&self) -> Result<Vec<DeviceInfo>, CaptureError> {
        // SPEC: wired after C1-C4 land: `crate::engine::devices::list_capture_devices()`.
        Err(CaptureError::Unexpected {
            stage: CaptureStage::Enumerate,
            detail: NOT_WIRED.to_string(),
        })
    }

    fn watch_devices(
        &self,
        tx: Sender<DeviceMsg>,
        dropped: Arc<AtomicU64>,
    ) -> Result<DeviceWatch, CaptureError> {
        // SPEC: wired after C1-C4 land: `crate::engine::devices::watch(tx, dropped)`.
        drop((tx, dropped));
        Err(CaptureError::Unexpected {
            stage: CaptureStage::Enumerate,
            detail: NOT_WIRED.to_string(),
        })
    }

    fn open_capture(
        &self,
        request: CaptureRequest,
    ) -> Result<(Worker, OpenedCapture), CaptureError> {
        // SPEC: wired after C1-C4 land: `crate::engine::capture::open(request)`.
        drop(request);
        Err(CaptureError::Unexpected {
            stage: CaptureStage::Spawn,
            detail: NOT_WIRED.to_string(),
        })
    }

    fn spawn_raw_input(&self, request: RawInputRequest) -> Result<Worker, VetoUnavailableReason> {
        // SPEC: wired after C1-C4 land: `crate::engine::rawinput::spawn(request)`.
        drop(request);
        Err(VetoUnavailableReason::ThreadFailed {
            message: NOT_WIRED.to_string(),
        })
    }

    fn platform(&self, observer: Arc<dyn OutputObserver>) -> Box<dyn PlatformActions> {
        let platform = crate::platform::windows::WindowsPlatform::new();
        // SPEC: wired after C1-C4 land: WP-C2 adds `set_output_observer`; the integration step
        // calls `platform.set_output_observer(observer)` here.
        drop(observer);
        Box::new(platform)
    }

    fn recorder_disk(&self) -> Arc<dyn RecorderDisk> {
        Arc::new(WindowsDisk)
    }
}

/// GetDiskFreeSpaceExW for the volume of a directory; writes are never stalled.
struct WindowsDisk;

impl RecorderDisk for WindowsDisk {
    fn free_bytes(&self, dir: &Path) -> std::io::Result<u64> {
        let wide: Vec<u16> = dir.as_os_str().encode_wide_nul();
        let mut free: u64 = 0;
        // SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the call, and `free` is a
        // valid out pointer; the two other outputs are not requested.
        unsafe { GetDiskFreeSpaceExW(PCWSTR(wide.as_ptr()), Some(&mut free), None, None) }
            .map_err(|e| std::io::Error::from_raw_os_error(e.code().0 & 0xFFFF))?;
        Ok(free)
    }

    fn before_write(&self) {}
}

/// `OsStr` → NUL-terminated UTF-16.
trait EncodeWideNul {
    fn encode_wide_nul(&self) -> Vec<u16>;
}

impl EncodeWideNul for std::ffi::OsStr {
    fn encode_wide_nul(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.encode_wide().chain(std::iter::once(0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_bytes_of_the_temp_volume_is_reported() {
        let free = WindowsDisk
            .free_bytes(&std::env::temp_dir())
            .expect("the temp volume reports its free space");
        assert!(free > 0);
    }

    #[test]
    fn timeout_until_is_zero_for_a_past_deadline() {
        let parts = WindowsParts::new();
        assert_eq!(parts.timeout_until(HostTime(0)), Duration::ZERO);
        let later = HostTime(parts.now().0 + 10_000_000);
        let wait = parts.timeout_until(later);
        assert!(wait > Duration::from_millis(900) && wait <= Duration::from_secs(1));
    }
}
