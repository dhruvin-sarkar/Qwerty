//! Every user-facing string (ui-v2 §9): the copy mechanics of U:1829-1842 here, one file of
//! messages per screen area below. Messages match on the engine enum that causes them, so each has
//! a single source and a missed variant fails to compile (U:1870).

pub mod diagnostics;
pub mod evaluation;
pub mod frame;
pub mod home;
pub mod settings;
pub mod status;
pub mod wizard;
pub mod zones;

/// The single-character ellipsis (U:1839).
pub const ELLIPSIS: char = '…';
/// The minus sign for negative values (U:1837).
pub const MINUS: char = '−';
/// The apostrophe (U:1838).
pub const APOSTROPHE: char = '’';
/// Quoted text in summaries is cut to this many characters plus "…" (U:689).
pub const QUOTE_LIMIT: usize = 24;

/// Straight quotes and apostrophes made curly (U:1838): every `'` becomes ’, and each `"` opens
/// (“) at the start or after whitespace or an opening bracket, else closes (”).
pub fn curly(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous: Option<char> = None;
    for c in text.chars() {
        let mapped = match c {
            '\'' => APOSTROPHE,
            '"' => {
                let opens = match previous {
                    None => true,
                    Some(p) => p.is_whitespace() || matches!(p, '(' | '[' | '{'),
                };
                if opens {
                    '“'
                } else {
                    '”'
                }
            }
            other => other,
        };
        out.push(mapped);
        previous = Some(c);
    }
    out
}

/// `text` cut to `limit` characters plus "…" when longer (U:1842); whitespace before the cut is
/// dropped so the ellipsis follows a character.
pub fn truncate_end(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let kept: String = text.chars().take(limit).collect();
    format!("{}{ELLIPSIS}", kept.trim_end())
}

/// User text quoted for a summary (U:689, U:1838): curly double quotes around the text cut to
/// [`QUOTE_LIMIT`] characters.
pub fn quote(text: &str) -> String {
    format!("“{}”", curly(&truncate_end(text, QUOTE_LIMIT)))
}

/// A command or progress label that asks for more input or is in progress: "Import…" (U:1839).
pub fn more(label: &str) -> String {
    format!("{label}{ELLIPSIS}")
}

/// A path shortened in the middle to at most `limit` characters (U:1842): the first two parts and
/// as many trailing parts as fit, around "…": `C:\Users\…\Spotify.exe`. A path whose ends alone
/// do not fit keeps its file name, cut at the end.
pub fn middle_path(path: &str, limit: usize) -> String {
    if path.chars().count() <= limit {
        return path.to_string();
    }
    let separator = if path.contains('\\') { "\\" } else { "/" };
    let parts: Vec<&str> = path.split(separator).collect();
    if parts.len() < 4 {
        // No middle part to drop.
        return truncate_end(path, limit);
    }
    let head = parts[..2].join(separator);
    // Separator, ellipsis, separator.
    let fixed = head.chars().count() + 3;
    // Trailing parts kept: at least the file name, never every part after the head.
    let mut kept = 1;
    while kept + 1 < parts.len() - 2 {
        let tail = parts[parts.len() - (kept + 1)..].join(separator);
        if fixed + tail.chars().count() > limit {
            break;
        }
        kept += 1;
    }
    let tail = parts[parts.len() - kept..].join(separator);
    let shortened = format!("{head}{separator}{ELLIPSIS}{separator}{tail}");
    if shortened.chars().count() <= limit {
        return shortened;
    }
    let file = parts[parts.len() - 1];
    let room = limit.saturating_sub(fixed).max(1);
    format!(
        "{head}{separator}{ELLIPSIS}{separator}{}",
        truncate_end(file, room)
    )
}

/// A counted phrase whose noun agrees with the number (U:1835): "1 tap", "10 taps".
pub fn count(n: u64, singular: &str, plural: &str) -> String {
    let noun = if n == 1 { singular } else { plural };
    format!("{n} {noun}")
}

/// A number with `decimals` places and the U+2212 minus sign (U:1834, U:1837): "−47".
pub fn number(value: f64, decimals: usize) -> String {
    let text = format!("{:.*}", decimals, value.abs());
    let zero = text.chars().all(|c| c == '0' || c == '.');
    if value < 0.0 && !zero {
        format!("{MINUS}{text}")
    } else {
        text
    }
}

/// A value with its unit after a space (U:1836): "48 kHz", "−52 dBFS".
pub fn with_unit(value: f64, decimals: usize, unit: &str) -> String {
    format!("{} {unit}", number(value, decimals))
}

/// A whole percentage with no space before "%" (U:1836): 0.94 → "94%". Rounds down, so a value
/// below a target never shows as the target.
pub fn percent(fraction: f64) -> String {
    format!("{}%", number((fraction * 100.0).floor(), 0))
}

/// Key names joined by "+" without spaces (U:1841): "Ctrl+Alt+Space".
pub fn keys(names: &[&str]) -> String {
    names.join("+")
}

/// The forms no copy string may contain (U:1827, U:1838-1839, U:1870), lower-case where the check
/// ignores case.
pub const BANNED: [&str; 8] = [
    "!",
    "please",
    "sorry",
    "oops",
    "click here",
    "...",
    "'",
    "\"",
];

/// The first banned form `text` contains, or a title that starts with "Failed" or "Error".
pub fn banned_form(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    if let Some(form) = BANNED.iter().find(|form| lower.contains(*form)) {
        return Some(form);
    }
    if text.starts_with("Failed") {
        return Some("Failed");
    }
    if text.starts_with("Error") {
        return Some("Error");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::engine::protocol::TelemetryFrame;
    use crate::shell::model::fixture;

    #[test]
    fn co1_copy_mechanics_on_their_examples() {
        // Counts (U:1835) and numerals (U:1834).
        assert_eq!(count(2, "zone", "zones"), "2 zones");
        assert_eq!(count(1, "minute", "minutes"), "1 minute");
        assert_eq!(count(4, "minute", "minutes"), "4 minutes");
        assert_eq!(count(1, "tap", "taps"), "1 tap");
        assert_eq!(count(10, "tap", "taps"), "10 taps");
        assert_eq!(count(0, "tap", "taps"), "0 taps");
        // Units (U:1836) and the minus sign (U:1837).
        assert_eq!(with_unit(48.0, 0, "kHz"), "48 kHz");
        assert_eq!(with_unit(38.0, 0, "ms"), "38 ms");
        assert_eq!(with_unit(-52.0, 0, "dBFS"), "−52 dBFS");
        assert_eq!(with_unit(-47.0, 0, "dBFS"), "−47 dBFS");
        assert_eq!(percent(0.94), "94%");
        assert_eq!(percent(0.949), "94%");
        assert_eq!(number(-0.0, 0), "0");
        assert_eq!(number(-0.04, 1), "0.0");
        assert_eq!(number(-3.25, 2), "−3.25");
        // Quotes and apostrophes (U:1838).
        assert_eq!(quote("Open Spotify"), "“Open Spotify”");
        assert_eq!(curly("it's"), "it’s");
        assert_eq!(curly("say \"hi\" (\"now\")"), "say “hi” (“now”)");
        // Ellipsis (U:1839).
        assert_eq!(more("Import"), "Import…");
        assert_eq!(more("Recalibrate"), "Recalibrate…");
        assert_eq!(more("Starting"), "Starting…");
        // Keys (U:1841).
        assert_eq!(keys(&["Ctrl", "Alt", "Space"]), "Ctrl+Alt+Space");
        assert_eq!(keys(&["Shift", "F6"]), "Shift+F6");
        assert_eq!(keys(&["Esc"]), "Esc");
        // Truncation (U:1842, U:689): 24 characters plus "…".
        assert_eq!(quote("Short"), "“Short”");
        assert_eq!(
            quote("Good morning, have a lovely day"),
            "“Good morning, have a lov…”"
        );
        assert_eq!(
            truncate_end("exactly twenty-four char", 24),
            "exactly twenty-four char"
        );
        assert_eq!(truncate_end("Good morning everyone", 13), "Good morning…");
        assert_eq!(
            middle_path("C:\\Users\\Me\\AppData\\Roaming\\Spotify\\Spotify.exe", 24),
            "C:\\Users\\…\\Spotify.exe"
        );
        assert_eq!(
            middle_path("C:\\Users\\Me\\AppData\\Roaming\\Spotify\\Spotify.exe", 32),
            "C:\\Users\\…\\Spotify\\Spotify.exe"
        );
        assert_eq!(middle_path("C:\\Tools\\run.exe", 40), "C:\\Tools\\run.exe");
        assert_eq!(
            middle_path("D:/QwertyData/DebugCaptures/20261111T084500Z/audio.wav", 30),
            "D:/QwertyData/…/audio.wav"
        );
    }

    #[test]
    fn co2_no_banned_form_in_copy_for_the_fixture_set() {
        let mut outputs = Vec::new();
        for text in fixture::texts() {
            outputs.push(quote(&text));
            outputs.push(curly(&text));
            outputs.push(more(&text));
            outputs.push(truncate_end(&text, QUOTE_LIMIT));
            outputs.push(middle_path(&text, 24));
        }
        for n in [0, 1, 2, 10] {
            outputs.push(count(n, "tap", "taps"));
        }
        let TelemetryFrame::Meter { rms, peak, noise } = fixture::meter_frame() else {
            panic!("the fixture's meter frame is a Meter");
        };
        for value in [rms, peak, noise] {
            let text = with_unit(value.get(), 0, "dBFS");
            assert!(text.starts_with(MINUS), "{text} uses U+2212");
            assert!(!text.contains('-'), "{text} has no hyphen-minus");
            outputs.push(text);
        }
        outputs.push(percent(fixture::overall().accuracy));
        for output in &outputs {
            assert_eq!(banned_form(output), None, "banned form in {output:?}");
        }
        // The check itself catches each form.
        for text in [
            "Done!",
            "Please wait",
            "Sorry",
            "Oops",
            "Click here",
            "Wait...",
            "it's",
            "\"x\"",
            "Failed to save",
            "Error",
        ] {
            assert!(banned_form(text).is_some(), "{text:?} is caught");
        }
    }
}
