//! LOG1 (p2 §2.9.4): action fields and file contents never reach the log. Its own binary,
//! because it installs a global subscriber.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use qwerty_app::engine::protocol::{
    AutostartPolicy, Command, DebugCaptureMode, Event, ProfileSummary,
};
use qwerty_app::engine::testkit::FakeOs;
use qwerty_app::engine::{Engine, EngineHandle, EngineOptions, EventStream};
use qwerty_app::platform::ActionError;
use qwerty_core::profile::ActionSpec;
use qwerty_core::store::{Profile, ZoneCount, ZoneEdit};

/// The string no log line may carry.
const SENTINEL: &str = "SENTINEL-7f3a9c";

/// A shared in-memory log.
#[derive(Clone, Default)]
struct Buffer(Arc<Mutex<Vec<u8>>>);

impl Write for Buffer {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log lock").extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Buffer {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("log lock")).into_owned()
    }
}

fn request(handle: &EngineHandle, command: Command) -> qwerty_app::engine::protocol::Reply {
    let text = format!("{:?}", command.kind());
    match handle.request(command) {
        Ok(reply) => reply,
        Err(e) => panic!("{text} failed: {e:?}"),
    }
}

/// Waits for `n` ActionCompleted events.
fn wait_completed(events: &EventStream, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut seen = 0;
    while seen < n {
        assert!(
            Instant::now() < deadline,
            "the test chains did not complete"
        );
        if let Some(Event::ActionCompleted { .. }) = events
            .recv_timeout(Duration::from_millis(100))
            .expect("the engine runs")
        {
            seen += 1;
        }
    }
}

#[test]
fn log1_no_action_field_or_file_content_reaches_the_log() {
    let log = Buffer::default();
    let writer = log.clone();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .init();

    let dir: PathBuf = std::env::temp_dir().join(format!("qwerty-log1-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("remove a stale temp dir");
    }
    // An unreadable profile whose JSON holds the sentinel, found by the startup scan.
    std::fs::create_dir_all(dir.join("Profiles")).expect("Profiles");
    std::fs::write(
        dir.join("Profiles").join("broken.json"),
        format!("{{\"schema_version\": 2, \"name\": \"{SENTINEL}\", oops"),
    )
    .expect("write the broken profile");

    let os = Arc::new(FakeOs::new(2.0));
    let (handle, events) = Engine::start_with(
        EngineOptions {
            data_dir: dir.clone(),
            autostart: AutostartPolicy::Untouched,
            debug_capture: DebugCaptureMode::FromConfig,
        },
        Arc::clone(&os) as Arc<dyn qwerty_app::engine::messages::OsParts>,
    )
    .expect("the engine starts");
    request(&handle, Command::ShellReady);

    // A profile to test chains on: imported, then made active.
    let file = dir.join("import.json");
    let profile = Profile::new(
        "Private".to_string(),
        ZoneCount::new(2).expect("2 zones"),
        chrono::Utc::now(),
    )
    .expect("a profile");
    qwerty_app::engine::persist::export_profile(&profile, &file).expect("export");
    let preview = request(&handle, Command::PreviewImport { path: file })
        .into_import_preview()
        .expect("ImportPreview");
    let zones: Vec<_> = preview.zones.iter().map(|z| z.id).collect();
    let id = request(
        &handle,
        Command::ImportProfile {
            preview: preview.preview,
        },
    )
    .into_profile_created()
    .expect("ProfileCreated");
    request(&handle, Command::SwitchProfile { id });
    let actions = [
        ActionSpec::RunCommand {
            command: format!("{SENTINEL}.exe"),
            args: vec![SENTINEL.to_string()],
        },
        ActionSpec::OpenUrl {
            url: format!("https://{SENTINEL}.invalid/"),
        },
    ];
    for (zone, action) in zones.iter().zip(actions) {
        request(
            &handle,
            Command::EditZone {
                zone: *zone,
                edit: ZoneEdit {
                    label: None,
                    layout: None,
                    trigger: None,
                    actions: Some(vec![action]),
                },
            },
        );
    }

    // Each chain's action fails with an error whose fields hold the sentinel.
    for zone in &zones {
        os.fail_next_action(ActionError::LaunchFailed {
            target: SENTINEL.to_string(),
            reason: SENTINEL.to_string(),
        });
        request(
            &handle,
            Command::TestChain {
                zone: *zone,
                delay: qwerty_core::units::Milliseconds::new(0.0),
            },
        );
        wait_completed(&events, 1);
    }

    let snapshot = handle.snapshot();
    let unreadable: Vec<&ProfileSummary> = snapshot
        .profiles
        .iter()
        .filter(|p| p.file == "broken.json")
        .collect();
    assert_eq!(unreadable.len(), 1, "the broken profile is listed");
    assert_eq!(os.platform_log().len(), 2, "both chains ran");
    handle.shutdown().expect("shutdown");
    std::fs::remove_dir_all(&dir).expect("remove the temp dir");

    let text = log.text();
    // The failures were logged, by file name and variant only.
    assert!(
        text.contains("broken.json"),
        "the unreadable profile was not logged"
    );
    assert!(
        text.contains("LaunchFailed"),
        "the action failures were not logged:\n{text}"
    );
    assert!(
        !text.contains(SENTINEL),
        "the log carries the sentinel:\n{}",
        text.lines()
            .filter(|l| l.contains(SENTINEL))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
