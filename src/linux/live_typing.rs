//! Labs › Live typing: a dictation's words appear in the focused field as
//! provisional text while you speak, through Wayland's input method protocol
//! (`zwp_input_method_v2`), and the finished text is committed there instead
//! of pasted. The app shows provisional text and replaces it each update, so
//! correcting a word needs no backspaces and can never touch another window.
//!
//! The protocol reaches a field only while its app has text-input enabled.
//! Anywhere else, and on X11 or a compositor without the protocol, nothing
//! shows while recording and the finished text pastes as usual.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, WrapErr, eyre};
use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::{wl_callback, wl_registry, wl_seat};
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols_misc::zwp_input_method_v2::client::{
    zwp_input_method_manager_v2::{self, ZwpInputMethodManagerV2},
    zwp_input_method_v2::{self, ZwpInputMethodV2},
};

use crate::dictation::live_text;
use crate::transcription::engine::Partial;

/// Below the protocol's 4000-byte limit on one string.
const MAX_TEXT_BYTES: usize = 3000;
/// How long after a failed connection the next one is tried: the capture loop
/// asks every turn, and another input method may leave the seat.
const RETRY: Duration = Duration::from_secs(30);

/// The focused field as the compositor last reported it. `done` applies what
/// the events before it announced, and every commit quotes how many there were.
#[derive(Default)]
struct Field {
    announced: bool,
    active: bool,
    done_events: u32,
    /// Finished texts sent, each of which ended the provisional text.
    commits: u32,
}

struct Events {
    field: Arc<Mutex<Field>>,
    unavailable: Arc<AtomicBool>,
}

impl Dispatch<ZwpInputMethodV2, ()> for Events {
    fn event(
        state: &mut Self,
        _: &ZwpInputMethodV2,
        event: zwp_input_method_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let mut field = state.field.lock().unwrap_or_else(PoisonError::into_inner);
        match event {
            zwp_input_method_v2::Event::Activate => field.announced = true,
            zwp_input_method_v2::Event::Deactivate => field.announced = false,
            zwp_input_method_v2::Event::Done => {
                field.active = field.announced;
                field.done_events = field.done_events.wrapping_add(1);
            }
            zwp_input_method_v2::Event::Unavailable => {
                field.active = false;
                state.unavailable.store(true, Ordering::Release);
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for Events {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for Events {
    fn event(
        _: &mut Self,
        _: &wl_callback::WlCallback,
        _: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for Events {
    fn event(
        _: &mut Self,
        _: &wl_seat::WlSeat,
        _: wl_seat::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwpInputMethodManagerV2, ()> for Events {
    fn event(
        _: &mut Self,
        _: &ZwpInputMethodManagerV2,
        _: zwp_input_method_manager_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

/// This session's input method. A thread reads the compositor's events;
/// requests go out from whichever thread calls.
pub(crate) struct LiveInput {
    connection: Connection,
    method: ZwpInputMethodV2,
    handle: QueueHandle<Events>,
    field: Arc<Mutex<Field>>,
    unavailable: Arc<AtomicBool>,
    closing: Arc<AtomicBool>,
}

impl Drop for LiveInput {
    /// Gives the seat back, for another input method, and wakes the reader
    /// thread so it ends.
    fn drop(&mut self) {
        self.closing.store(true, Ordering::Release);
        self.method.destroy();
        let _ = self.connection.display().sync(&self.handle, ());
        let _ = self.connection.flush();
    }
}

impl LiveInput {
    fn connect() -> Result<Self> {
        let connection =
            Connection::connect_to_env().wrap_err("could not reach the Wayland compositor")?;
        let (globals, mut queue) = registry_queue_init::<Events>(&connection)
            .wrap_err("could not read the compositor's globals")?;
        let handle = queue.handle();
        let manager: ZwpInputMethodManagerV2 = globals
            .bind(&handle, 1..=1, ())
            .wrap_err("the compositor offers no input method protocol")?;
        let seat: wl_seat::WlSeat = globals
            .bind(&handle, 1..=1, ())
            .wrap_err("the compositor offers no seat")?;
        let method = manager.get_input_method(&seat, &handle, ());
        let field = Arc::new(Mutex::new(Field::default()));
        let unavailable = Arc::new(AtomicBool::new(false));
        let mut events = Events {
            field: field.clone(),
            unavailable: unavailable.clone(),
        };
        queue
            .roundtrip(&mut events)
            .wrap_err("the compositor did not answer")?;
        if unavailable.load(Ordering::Acquire) {
            return Err(eyre!("another input method holds this seat"));
        }
        let closing = Arc::new(AtomicBool::new(false));
        let lost = unavailable.clone();
        let reader_closing = closing.clone();
        thread::Builder::new()
            .name("hex-live-typing".into())
            .spawn(move || {
                while !reader_closing.load(Ordering::Acquire)
                    && queue.blocking_dispatch(&mut events).is_ok()
                {}
                lost.store(true, Ordering::Release);
            })
            .wrap_err("could not start the input method thread")?;
        Ok(Self {
            connection,
            method,
            handle,
            field,
            unavailable,
            closing,
        })
    }

    /// Sends `requests` and the commit that applies them, when a field is
    /// focused; false when none is or the connection failed.
    fn send(&self, requests: impl FnOnce(&ZwpInputMethodV2, u32)) -> bool {
        let field = self.field();
        if !field.active || self.unavailable.load(Ordering::Acquire) {
            return false;
        }
        requests(&self.method, field.done_events);
        self.connection.flush().is_ok()
    }

    fn field(&self) -> std::sync::MutexGuard<'_, Field> {
        self.field.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Changes whenever provisional text may have left the field: a focus
    /// change or a finished text.
    fn epoch(&self) -> u32 {
        let field = self.field();
        field.done_events.wrapping_add(field.commits)
    }

    fn show(&self, text: &str) -> bool {
        let shown = last_bytes(text, MAX_TEXT_BYTES);
        let cursor = shown.len() as i32;
        self.send(|method, serial| {
            method.set_preedit_string(shown.to_owned(), cursor, cursor);
            method.commit(serial);
        })
    }

    // A commit without a preedit clears the provisional text.
    fn clear(&self) {
        self.send(|method, serial| method.commit(serial));
    }

    /// Types `text` into the focused field in place of the provisional text.
    /// False leaves the text to the paster: without a field, and for a line
    /// break, which a terminal would run where a paste stays bracketed.
    pub(crate) fn type_final(&self, text: &str) -> bool {
        if text.contains(['\n', '\r']) {
            // Out of the paste's way, which some apps would add to it.
            self.clear();
            return false;
        }
        let typed = self.send(|method, serial| {
            for piece in pieces(text, MAX_TEXT_BYTES) {
                method.commit_string(piece.to_owned());
                method.commit(serial);
            }
        });
        if typed {
            let mut field = self.field();
            field.commits = field.commits.wrapping_add(1);
        }
        typed
    }
}

struct Held {
    input: Option<Arc<LiveInput>>,
    failed_at: Option<Instant>,
}

static HELD: Mutex<Held> = Mutex::new(Held {
    input: None,
    failed_at: None,
});

fn held() -> std::sync::MutexGuard<'static, Held> {
    HELD.lock().unwrap_or_else(PoisonError::into_inner)
}

fn may_connect(failed_at: Option<Instant>, now: Instant) -> bool {
    failed_at.is_none_or(|failed_at| now.duration_since(failed_at) >= RETRY)
}

/// The session's input method while Labs › Live typing is on, connected at
/// the first use and again after a lost connection, at most every [`RETRY`].
pub(crate) fn input() -> Option<Arc<LiveInput>> {
    if !crate::labs::live_typing() {
        return None;
    }
    let mut held = held();
    if held
        .input
        .as_ref()
        .is_some_and(|input| input.unavailable.load(Ordering::Acquire))
    {
        held.input = None;
    }
    if held.input.is_none() && may_connect(held.failed_at, Instant::now()) {
        match LiveInput::connect() {
            Ok(connected) => {
                held.input = Some(Arc::new(connected));
                held.failed_at = None;
            }
            Err(error) => {
                held.failed_at = Some(Instant::now());
                tracing::warn!(error = %format!("{error:#}"), "live typing is unavailable");
            }
        }
    }
    held.input.clone()
}

/// The preview chunk for a dictation starting now: only with a field that can
/// show its words. Without one the dictation streams and pastes as usual.
pub(crate) fn preview_chunk_ms() -> Option<u32> {
    let input = input()?;
    let active = input.field().active;
    active.then(crate::labs::live_typing_chunk_ms)
}

/// The capture loop's side: keeps the field's provisional text on the
/// recording's latest words, and past the release until the finished text
/// replaces it, so the words never blink out while the last ones transcribe.
#[derive(Default)]
pub(crate) struct LiveTyping {
    shown: Option<(String, u32)>,
}

impl LiveTyping {
    /// Call once per capture loop turn. `previewing` is a recording with a
    /// preview stream; `finishing` holds the text while a finished
    /// recording's output is on its way. Turning the lab off clears the text
    /// and gives the seat back.
    pub(crate) fn update(&mut self, previewing: bool, finishing: bool) {
        if !crate::labs::live_typing() {
            self.clear();
            held().input = None;
        } else if previewing {
            let Some(text) = live_text::current().as_ref().map(spoken) else {
                return;
            };
            let Some(input) = input().filter(|_| !text.is_empty()) else {
                return;
            };
            let epoch = input.epoch();
            if self
                .shown
                .as_ref()
                .is_some_and(|shown| shown.0 == text && shown.1 == epoch)
            {
                return;
            }
            if input.show(&text) {
                self.shown = Some((text, epoch));
            }
        } else if !finishing {
            self.clear();
        }
    }

    /// Removes the text after a short tap, a cancel, or a failure; harmless
    /// after a commit.
    fn clear(&mut self) {
        if self.shown.take().is_none() {
            return;
        }
        let input = held().input.clone();
        if let Some(input) = input {
            input.clear();
        }
    }
}

fn spoken(partial: &Partial) -> String {
    format!("{}{}", partial.stable, partial.tentative)
        .trim()
        .to_owned()
}

/// The longest end of `text` within `limit` bytes, on a character boundary:
/// a long dictation keeps showing its newest words.
fn last_bytes(text: &str, limit: usize) -> &str {
    let mut start = text.len().saturating_sub(limit);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn pieces(text: &str, limit: usize) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let mut end = rest.len().min(limit);
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        let (piece, tail) = rest.split_at(end);
        pieces.push(piece);
        rest = tail;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "needs a compositor offering the input method protocol"]
    fn the_compositor_accepts_the_input_method() {
        let input = LiveInput::connect().unwrap();
        assert!(!input.unavailable.load(Ordering::Acquire));
        println!("field active: {}", input.field().active);
    }

    #[test]
    fn the_shown_text_keeps_the_newest_words_and_whole_characters() {
        assert_eq!(last_bytes("short", 10), "short");
        assert_eq!(last_bytes("abcdef", 3), "def");
        assert_eq!(last_bytes("aééb", 4), "éb");
    }

    #[test]
    fn a_failed_connection_waits_before_the_next_try() {
        let now = Instant::now();
        assert!(may_connect(None, now));
        assert!(!may_connect(Some(now), now + RETRY / 2));
        assert!(may_connect(Some(now), now + RETRY));
    }

    #[test]
    fn a_long_text_splits_into_whole_characters_in_order() {
        assert_eq!(pieces("abcde", 2), ["ab", "cd", "e"]);
        assert_eq!(pieces("ééé", 3), ["é", "é", "é"]);
        assert!(pieces("", 3).is_empty());
    }

    #[test]
    fn spoken_text_is_settled_words_then_the_rest() {
        let partial = Partial {
            stable: "so the".into(),
            tentative: " plan ".into(),
        };
        assert_eq!(spoken(&partial), "so the plan");
    }
}
