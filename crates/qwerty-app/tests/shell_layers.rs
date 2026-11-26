//! SB1 (shell-v2 §1.2, §4.2 WP-S0): the shell's layer rule, checked on the source. `model` and
//! `win` name no GUI framework; `model` names no Windows API, tray, hotkey or toast crate and no
//! `platform` item other than `ActionError`; nothing in the shell names a v1 module (MV:217).

use std::path::{Path, PathBuf};

fn shell_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("shell")
}

/// Every `.rs` file under `dir`, recursively.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("a readable source directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The source with `//` comments removed, so prose may name what code may not.
fn code(text: &str) -> String {
    text.lines()
        .map(|line| match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Every place `word` appears as a whole identifier, with the text after it.
fn whole_words<'a>(text: &'a str, word: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(found) = text[from..].find(word) {
        let start = from + found;
        let end = start + word.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        if !before.is_some_and(is_ident) && !after.is_some_and(is_ident) {
            out.push(&text[end..]);
        }
        from = end;
    }
    out
}

/// `name` used as a path root or imported: `name::…`, `use name`, `extern crate name`.
fn names_crate(code: &str, name: &str) -> bool {
    whole_words(code, name)
        .iter()
        .any(|rest| rest.starts_with("::"))
        || code.contains(&format!("use {name};"))
        || code.contains(&format!("use {name} "))
        || code.contains(&format!("extern crate {name}"))
}

/// Every item named after `crate::platform::`, including a `{…}` group.
fn platform_items(code: &str) -> Vec<String> {
    let mut items = Vec::new();
    let marker = "crate::platform::";
    let mut from = 0;
    while let Some(found) = code[from..].find(marker) {
        let rest = &code[from + found + marker.len()..];
        if let Some(group) = rest.strip_prefix('{') {
            let inner = &group[..group.find('}').expect("a closed import group")];
            items.extend(
                inner
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            );
        } else {
            items.push(rest.chars().take_while(|c| is_ident(*c)).collect());
        }
        from += found + marker.len();
    }
    items
}

/// Bare `crate::platform` (a module import) counts as naming more than `ActionError`.
fn names_platform_module(code: &str) -> bool {
    whole_words(code, "crate::platform")
        .iter()
        .any(|rest| !rest.starts_with("::"))
}

/// The shell layer `layer` named as a path: `shell::win`, or relative `super::win`.
fn names_shell_layer(code: &str, layer: &str) -> bool {
    ["shell", "super"]
        .iter()
        .any(|parent| !whole_words(code, &format!("{parent}::{layer}")).is_empty())
}

const V1_MODULES: [&str; 6] = [
    "app_state",
    "tray",
    "hotkeys",
    "ui",
    "audio_thread",
    "capture_worker",
];
const V1_FUNCTIONS: [&str; 3] = [
    "run_action_chain",
    "platform_for_this_os",
    "set_start_with_windows",
];

/// The layer violations in one file's code.
fn violations(relative: &str, code: &str) -> Vec<String> {
    let mut found = Vec::new();
    let in_model = relative.starts_with("model");
    let in_win = relative.starts_with("win");
    if in_model || in_win {
        for framework in ["rune", "gpui"] {
            if names_crate(code, framework) {
                found.push(format!("{relative}: names the GUI framework `{framework}`"));
            }
        }
    }
    if in_model {
        for name in ["windows", "tray_icon", "global_hotkey", "winrt_toast"] {
            if names_crate(code, name) {
                found.push(format!("{relative}: names `{name}`"));
            }
        }
        for item in platform_items(code) {
            if item != "ActionError" {
                found.push(format!("{relative}: names `crate::platform::{item}`"));
            }
        }
        if names_platform_module(code) {
            found.push(format!("{relative}: imports the `crate::platform` module"));
        }
    }
    // The shell layers a file may not use (§1.2: arrows point from VIEW to WINDOWS to MODEL).
    let above: &[&str] = if in_model {
        &["win", "view"]
    } else if in_win {
        &["view"]
    } else {
        &[]
    };
    for layer in above {
        if names_shell_layer(code, layer) {
            found.push(format!(
                "{relative}: uses the shell layer `{layer}` above it"
            ));
        }
    }
    for module in V1_MODULES {
        if !whole_words(code, &format!("crate::{module}")).is_empty() {
            found.push(format!("{relative}: names the v1 module `crate::{module}`"));
        }
    }
    for function in V1_FUNCTIONS {
        if !whole_words(code, function).is_empty() {
            found.push(format!("{relative}: names the v1 item `{function}`"));
        }
    }
    found
}

#[test]
fn sb1_shell_layers_hold() {
    let root = shell_dir();
    let files = sources(&root);
    // Every stub of §4.2 is in place, so the scan covers the whole planned tree.
    assert!(files.len() >= 50, "only {} shell files found", files.len());

    let mut found = Vec::new();
    for file in &files {
        let relative = file
            .strip_prefix(&root)
            .expect("under the shell directory")
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(file).expect("a UTF-8 source file");
        found.extend(violations(&relative, &code(&text)));
    }
    assert!(found.is_empty(), "layer violations:\n{}", found.join("\n"));

    // The checks catch what they look for.
    let planted = [
        ("model/a.rs", "use rune::El;"),
        ("win/a.rs", "let x = gpui::App::new();"),
        ("model/a.rs", "use windows::Win32::Foundation::HWND;"),
        ("model/a.rs", "use tray_icon::TrayIcon;"),
        ("model/a.rs", "global_hotkey::GlobalHotKeyManager::new()"),
        ("model/a.rs", "use winrt_toast::Toast;"),
        (
            "model/a.rs",
            "use crate::platform::{ActionError, PlatformActions};",
        ),
        ("model/a.rs", "use crate::platform::ToastSpec;"),
        ("model/a.rs", "use crate::platform;"),
        ("view/a.rs", "use crate::app_state::AppState;"),
        ("win/a.rs", "crate::tray::build()"),
        ("view/a.rs", "run_action_chain(&chain)"),
        ("view/a.rs", "set_start_with_windows(true)"),
        ("view/a.rs", "platform_for_this_os()"),
        // Arrows point down only (§1.2): the model uses neither shell layer above it, and the
        // windows layer does not use the view.
        ("model/a.rs", "use crate::shell::win::tray;"),
        ("model/a.rs", "use qwerty_app::shell::win::dwm::apply;"),
        ("model/vm/a.rs", "use super::super::win::system;"),
        ("model/a.rs", "use crate::shell::view::app;"),
        ("win/a.rs", "use crate::shell::view::theme;"),
        ("win/a.rs", "use super::view::app;"),
    ];
    for (relative, line) in planted {
        assert!(
            !violations(relative, line).is_empty(),
            "{line:?} in {relative} is caught"
        );
    }
    let allowed = [
        ("model/a.rs", "use crate::platform::ActionError;"),
        ("view/a.rs", "use crate::shell::win::tray;"),
        ("win/a.rs", "use crate::shell::model::bridge::TrayCommand;"),
        ("view/a.rs", "use crate::shell::model::store::Store;"),
        ("win/a.rs", "use windows::Win32::Foundation::HWND;"),
        ("view/a.rs", "use rune::El;"),
        ("model/a.rs", "let runes = 1; let windows_open = 2;"),
    ];
    for (relative, line) in allowed {
        assert_eq!(violations(relative, line), Vec::<String>::new(), "{line:?}");
    }
}
