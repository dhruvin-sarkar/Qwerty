//! The request worker (shell-v2 §2.1, §2.3): one `qwerty-shell-requests` thread runs the shell's
//! commands one at a time, FIFO, so the engine sees them in the order the user made them (MR:57),
//! and posts each reply into the inbox. The UI thread only submits: it never calls `request`
//! (B:9, MB:275). The UI-thread side keeps one in-flight flag per outstanding request and the
//! latest reply error per command kind, until the next reply of that kind (§1.3, AB-12(a)).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};

use crate::engine::protocol::{Command, CommandKind, EngineError, Reply};
use crate::shell::model::bridge::{Inbox, ShellInput};

/// Identifies one submitted command and its reply. Increasing in submission order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

/// Runs one command against the engine: `EngineHandle::request` (and, for `Shutdown`,
/// `EngineHandle::shutdown`), supplied by the view so tests need no engine.
pub type Runner = Box<dyn FnMut(Command) -> Result<Reply, EngineError> + Send>;

/// The OS name of the worker thread (§2.1).
pub const REQUESTS_THREAD: &str = "qwerty-shell-requests";

struct Job {
    id: RequestId,
    command: Command,
}

/// The worker and its UI-thread bookkeeping. Dropping it joins the thread after the request that
/// is running; requests still queued are not sent.
pub struct Requests {
    jobs: Option<Sender<Job>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    inbox: Arc<Inbox>,
    next: u64,
    in_flight: BTreeMap<RequestId, CommandKind>,
    errors: HashMap<CommandKind, EngineError>,
}

impl Requests {
    /// Spawns the worker. Each reply is posted as `ShellInput::Reply{id, result}`.
    pub fn spawn(run: Runner, inbox: Arc<Inbox>) -> std::io::Result<Requests> {
        let (jobs, queue) = crossbeam_channel::unbounded();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let stop = Arc::clone(&stop);
            let inbox = Arc::clone(&inbox);
            std::thread::Builder::new()
                .name(REQUESTS_THREAD.to_string())
                .spawn(move || work(run, &queue, &stop, &inbox))?
        };
        Ok(Requests {
            jobs: Some(jobs),
            stop,
            thread: Some(thread),
            inbox,
            next: 0,
            in_flight: BTreeMap::new(),
            errors: HashMap::new(),
        })
    }

    /// Queues `command` and sets its in-flight flag; returns at once. A worker that is gone (its
    /// runner panicked) answers at once with `EngineStopped`, through the inbox like any reply.
    pub fn submit(&mut self, command: Command) -> RequestId {
        let id = RequestId(self.next);
        self.next += 1;
        self.in_flight.insert(id, command.kind());
        let sent = match &self.jobs {
            Some(jobs) => jobs.send(Job { id, command }).is_ok(),
            None => false,
        };
        if !sent {
            self.inbox.push(ShellInput::Reply {
                id,
                result: Err(EngineError::EngineStopped),
            });
        }
        id
    }

    /// Applies a reply from the inbox: clears its in-flight flag and records (on an error) or
    /// clears (on success) the latest error of its command kind. Returns the request's kind, or
    /// `None` when `id` was not in flight.
    pub fn complete(
        &mut self,
        id: RequestId,
        result: &Result<Reply, EngineError>,
    ) -> Option<CommandKind> {
        let kind = self.in_flight.remove(&id)?;
        match result {
            Ok(_) => {
                self.errors.remove(&kind);
            }
            Err(error) => {
                self.errors.insert(kind, error.clone());
            }
        }
        Some(kind)
    }

    /// The in-flight flag of one request.
    pub fn is_in_flight(&self, id: RequestId) -> bool {
        self.in_flight.contains_key(&id)
    }

    /// Whether any request of `kind` is in flight.
    pub fn kind_in_flight(&self, kind: CommandKind) -> bool {
        self.in_flight.values().any(|k| *k == kind)
    }

    /// The latest reply error of `kind`, until the next reply of that kind.
    pub fn last_error(&self, kind: CommandKind) -> Option<&EngineError> {
        self.errors.get(&kind)
    }
}

impl Drop for Requests {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            // A runner that panicked has already ended the thread; there is nothing to report
            // during teardown.
            drop(thread.join());
        }
    }
}

fn work(mut run: Runner, queue: &Receiver<Job>, stop: &AtomicBool, inbox: &Inbox) {
    while let Ok(job) = queue.recv() {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let result = run(job.command);
        inbox.push(ShellInput::Reply { id: job.id, result });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    fn inbox() -> Arc<Inbox> {
        Arc::new(Inbox::new(Box::new(|| {})))
    }

    /// Waits until the inbox has `n` inputs, collecting them.
    fn replies(inbox: &Inbox, n: usize) -> Vec<(RequestId, Result<Reply, EngineError>)> {
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while out.len() < n && Instant::now() < deadline {
            for input in inbox.drain() {
                let ShellInput::Reply { id, result } = input else {
                    panic!("the worker posts only replies");
                };
                out.push((id, result));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(out.len(), n, "every reply arrived");
        out
    }

    #[test]
    fn rq1_commands_complete_in_submission_order_with_their_ids() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let inbox = inbox();
        let mut requests = Requests::spawn(
            Box::new(move |command: Command| {
                // Later commands take less time, so only a FIFO worker keeps the order.
                let n = record.lock().expect("test lock").len() as u64;
                std::thread::sleep(Duration::from_millis(10 * (5 - n.min(5))));
                record.lock().expect("test lock").push(command.kind());
                Ok(Reply::Ack)
            }),
            Arc::clone(&inbox),
        )
        .expect("spawns");
        let commands = vec![
            Command::ShellReady,
            Command::ListDevices,
            Command::SetListening { listening: true },
            Command::RetryCapture,
            Command::GetInspection,
        ];
        let kinds: Vec<CommandKind> = commands.iter().map(Command::kind).collect();
        let ids: Vec<RequestId> = commands.into_iter().map(|c| requests.submit(c)).collect();
        let got = replies(&inbox, 5);
        let got_ids: Vec<RequestId> = got.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            got_ids, ids,
            "replies in submission order, each with its id"
        );
        assert_eq!(*seen.lock().expect("test lock"), kinds, "sent in order");
        for (id, result) in &got {
            assert_eq!(requests.complete(*id, result), Some(kinds[id.0 as usize]));
        }
    }

    #[test]
    fn rq2_in_flight_flag_is_set_on_submit_and_cleared_on_any_reply() {
        let inbox = inbox();
        let failed = AtomicBool::new(false);
        let mut requests = Requests::spawn(
            Box::new(move |command: Command| {
                // The first RetryCapture fails; everything else succeeds.
                if command == Command::RetryCapture && !failed.swap(true, Ordering::SeqCst) {
                    Err(EngineError::NoActiveProfile)
                } else {
                    Ok(Reply::Ack)
                }
            }),
            Arc::clone(&inbox),
        )
        .expect("spawns");

        let failing = requests.submit(Command::RetryCapture);
        assert!(requests.is_in_flight(failing));
        assert!(requests.kind_in_flight(CommandKind::RetryCapture));
        let (id, result) = replies(&inbox, 1).remove(0);
        assert_eq!(id, failing);
        assert_eq!(
            requests.complete(id, &result),
            Some(CommandKind::RetryCapture)
        );
        assert!(!requests.is_in_flight(failing), "cleared by an error reply");
        assert!(!requests.kind_in_flight(CommandKind::RetryCapture));
        assert_eq!(
            requests.last_error(CommandKind::RetryCapture),
            Some(&EngineError::NoActiveProfile)
        );

        // A success of another kind leaves that error; the next reply of its own kind clears it.
        let other = requests.submit(Command::ShellReady);
        assert!(requests.is_in_flight(other));
        let (id, result) = replies(&inbox, 1).remove(0);
        requests.complete(id, &result);
        assert!(!requests.is_in_flight(other), "cleared by a success reply");
        assert!(requests.last_error(CommandKind::RetryCapture).is_some());
        assert_eq!(requests.complete(id, &result), None, "already completed");
        let again = requests.submit(Command::RetryCapture);
        let (id, _) = replies(&inbox, 1).remove(0);
        assert_eq!(id, again);
        requests.complete(id, &Ok(Reply::Ack));
        assert_eq!(
            requests.last_error(CommandKind::RetryCapture),
            None,
            "the next success of the same kind clears it"
        );
    }

    #[test]
    fn rq3_dropping_the_worker_joins_after_the_current_request() {
        let started = Arc::new(AtomicBool::new(false));
        let ran = Arc::new(Mutex::new(0u32));
        let inbox = inbox();
        let mut requests = {
            let started = Arc::clone(&started);
            let ran = Arc::clone(&ran);
            Requests::spawn(
                Box::new(move |_| {
                    started.store(true, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(200));
                    *ran.lock().expect("test lock") += 1;
                    Ok(Reply::Ack)
                }),
                Arc::clone(&inbox),
            )
            .expect("spawns")
        };
        for _ in 0..3 {
            requests.submit(Command::ListDevices);
        }
        while !started.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let start = Instant::now();
        drop(requests);
        let took = start.elapsed();
        assert!(
            took < Duration::from_millis(200 + 500),
            "joined {took:?} after the drop"
        );
        assert_eq!(
            *ran.lock().expect("test lock"),
            1,
            "queued requests are not sent"
        );
        assert_eq!(inbox.drain().len(), 1, "the current request still replies");
    }
}
