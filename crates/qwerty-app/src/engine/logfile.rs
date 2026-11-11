//! Rotating log file (p2 §2.9.4, PARITY-36). `RotatingLog` is a `MakeWriter` for
//! `Logs/qwerty.log`: once a record would take the file past `LOG_FILE_LIMIT`, the file is renamed
//! to `qwerty.1.log` (replacing the previous one) and a new one starts, so logs never exceed twice
//! the limit. `main.rs` installs it with the subscriber; the engine never installs one.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use tracing_subscriber::fmt::MakeWriter;

use crate::engine::protocol::LOG_FILE_LIMIT;

/// The live log file in the logs folder.
pub const LOG_FILE: &str = "qwerty.log";
/// The previous log file, replaced at every rotation.
pub const ROTATED_LOG_FILE: &str = "qwerty.1.log";

pub struct RotatingLog {
    dir: PathBuf,
    current: Mutex<CurrentFile>,
}

struct CurrentFile {
    file: File,
    /// Bytes in `file`, so rotation needs no metadata call per record.
    len: u64,
}

impl RotatingLog {
    /// create_dir_all(logs_dir), then opens Logs/qwerty.log for append.
    pub fn new(logs_dir: &Path) -> io::Result<RotatingLog> {
        std::fs::create_dir_all(logs_dir)?;
        let file = open_append(&logs_dir.join(LOG_FILE))?;
        let len = file.metadata()?.len();
        Ok(RotatingLog {
            dir: logs_dir.to_path_buf(),
            current: Mutex::new(CurrentFile { file, len }),
        })
    }

    fn lock(&self) -> io::Result<MutexGuard<'_, CurrentFile>> {
        self.current
            .lock()
            .map_err(|_| io::Error::other("a log write panicked while it held the log file"))
    }

    /// Writes one formatted record whole, rotating first if it would take a non-empty file past
    /// the limit, so a record is never split across the two files.
    fn write_record(&self, record: &[u8]) -> io::Result<()> {
        let mut current = self.lock()?;
        let len = record.len() as u64;
        if current.len > 0 && current.len + len > LOG_FILE_LIMIT {
            // std opens files with FILE_SHARE_DELETE, so the open file can be renamed.
            std::fs::rename(self.dir.join(LOG_FILE), self.dir.join(ROTATED_LOG_FILE))?;
            *current = CurrentFile {
                file: open_append(&self.dir.join(LOG_FILE))?,
                len: 0,
            };
        }
        current.file.write_all(record)?;
        current.len += len;
        Ok(())
    }
}

fn open_append(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

/// One `MakeWriter` handle; each `write` is one formatted record.
pub struct LogWriter<'a> {
    log: &'a RotatingLog,
}

impl Write for LogWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.log.write_record(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.log.lock()?.file.flush()
    }
}

impl<'a> MakeWriter<'a> for RotatingLog {
    type Writer = LogWriter<'a>;

    fn make_writer(&'a self) -> LogWriter<'a> {
        LogWriter { log: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory under the temp dir, removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let dir =
                std::env::temp_dir().join(format!("qwerty-c4-log-{name}-{}", std::process::id()));
            if dir.exists() {
                std::fs::remove_dir_all(&dir).expect("clear the old test directory");
            }
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Cleanup only: a leftover temp folder fails nothing.
            if let Err(e) = std::fs::remove_dir_all(&self.0) {
                eprintln!("could not remove {}: {e}", self.0.display());
            }
        }
    }

    fn size(path: &Path) -> u64 {
        std::fs::metadata(path).expect("the file exists").len()
    }

    #[test]
    fn rotates_once_a_record_would_pass_the_limit() {
        let tmp = TempDir::new("rotate");
        let logs = tmp.0.join("Logs");
        let log = RotatingLog::new(&logs).expect("opens");
        let live = logs.join(LOG_FILE);
        let rotated = logs.join(ROTATED_LOG_FILE);
        assert!(live.is_file() && !rotated.exists());

        let record = vec![b'a'; 64 * 1024];
        let per_file = LOG_FILE_LIMIT / record.len() as u64;
        for _ in 0..per_file {
            log.make_writer().write_all(&record).expect("writes");
        }
        assert_eq!(size(&live), LOG_FILE_LIMIT);
        assert!(!rotated.exists(), "exactly the limit does not rotate");

        log.make_writer().write_all(b"next\n").expect("writes");
        assert_eq!(size(&rotated), LOG_FILE_LIMIT);
        assert_eq!(std::fs::read(&live).expect("reads"), b"next\n");

        // A second rotation replaces qwerty.1.log, so the folder never holds more than two files.
        for _ in 0..per_file {
            log.make_writer().write_all(&record).expect("writes");
        }
        assert_eq!(size(&rotated), LOG_FILE_LIMIT - record.len() as u64 + 5);
        assert_eq!(size(&live), record.len() as u64);
        drop(log);

        // Reopening appends to the live file and keeps counting its bytes.
        let log = RotatingLog::new(&logs).expect("reopens");
        log.make_writer().write_all(b"more\n").expect("writes");
        log.make_writer().flush().expect("flushes");
        assert_eq!(size(&live), record.len() as u64 + 5);
    }
}
