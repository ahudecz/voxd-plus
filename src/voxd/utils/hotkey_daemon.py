"""Global hotkey daemon using evdev.

Captures a configurable key pattern (double-tap, hold, or single press)
directly from /dev/input/event* and sends 'trigger_record' to the
running VOXD instance via its Unix socket IPC.

Requires membership in the 'input' group (same as ydotool).

Usage:
    voxd-plus --hotkeyd              # foreground
    voxd-plus --hotkeyd --daemonize  # background (nohup)
"""

import asyncio
import os
import socket
import time
from pathlib import Path
from typing import Optional

from voxd.utils.libw import verbo, verr


# ---------------------------------------------------------------------------
# Key name → evdev scancode mapping (common trigger keys)
# ---------------------------------------------------------------------------

_KEY_CODES: dict[str, int] = {
    "KEY_CAPSLOCK": 58,
    "KEY_LEFTCTRL": 29,
    "KEY_RIGHTCTRL": 97,
    "KEY_LEFTSHIFT": 42,
    "KEY_RIGHTSHIFT": 54,
    "KEY_LEFTMETA": 125,
    "KEY_RIGHTMETA": 126,
    "KEY_RIGHTALT": 100,
    "KEY_LEFTALT": 56,
    "KEY_F13": 183,
    "KEY_F14": 184,
    "KEY_F15": 185,
    "KEY_F16": 186,
    "KEY_F17": 187,
    "KEY_F18": 188,
    "KEY_SCROLLLOCK": 70,
    "KEY_PAUSE": 119,
    "KEY_PAGEDOWN": 109,
    "KEY_PAGEUP": 104,
    "KEY_INSERT": 110,
}


def _resolve_key_code(key_name: str) -> int:
    """Resolve a key name to its evdev scancode."""
    upper = key_name.upper()
    if not upper.startswith("KEY_"):
        upper = "KEY_" + upper
    if upper in _KEY_CODES:
        return _KEY_CODES[upper]
    # Try importing evdev for full keycode table
    try:
        import evdev.ecodes as ec
        code = getattr(ec, upper, None)
        if code is not None:
            return int(code)
    except ImportError:
        pass
    raise ValueError(f"Unknown key name: {key_name}. Available: {', '.join(sorted(_KEY_CODES.keys()))}")


class HotkeyDaemon:
    """Global hotkey listener using evdev async reads."""

    def __init__(
        self,
        trigger_key: str = "KEY_CAPSLOCK",
        trigger_key_modifier: str = "",
        trigger_key_2: str = "",
        trigger_key_2_modifier: str = "",
        trigger_key_2_lang: str = "hu",
        mode: str = "double_tap",
        double_tap_window_ms: int = 350,
        hold_threshold_ms: int = 300,
        suppress_original: bool = True,
        socket_path: Optional[str] = None,
    ):
        self.trigger_key = trigger_key
        self.key_code = _resolve_key_code(trigger_key)
        self.trigger_key_modifier = trigger_key_modifier
        self.modifier_code: Optional[int] = None
        if trigger_key_modifier:
            try:
                self.modifier_code = _resolve_key_code(trigger_key_modifier)
            except ValueError as e:
                verr(f"[hotkeyd] Modifier key invalid: {e}")
        self.trigger_key_2 = trigger_key_2
        self.trigger_key_2_modifier = trigger_key_2_modifier
        self.modifier_code_2: Optional[int] = None
        if trigger_key_2_modifier:
            try:
                self.modifier_code_2 = _resolve_key_code(trigger_key_2_modifier)
            except ValueError as e:
                verr(f"[hotkeyd] Modifier key 2 invalid: {e}")
        self.trigger_key_2_lang = trigger_key_2_lang
        self.key_code_2: Optional[int] = None
        if trigger_key_2:
            try:
                self.key_code_2 = _resolve_key_code(trigger_key_2)
            except ValueError as e:
                verr(f"[hotkeyd] Second trigger key invalid: {e}")
        self.mode = mode
        self.double_tap_window = double_tap_window_ms / 1000.0
        self.hold_threshold = hold_threshold_ms / 1000.0
        self.suppress_original = suppress_original
        self.socket_path = socket_path or str(
            Path.home() / ".config" / "voxd-plus" / "voxd-plus.sock"
        )

        # State
        self._last_tap_time: float = 0.0
        self._hold_start: float = 0.0
        self._running = False
        # Modifier key held state (shared across devices)
        self._modifier_held: set[int] = set()

    def run(self):
        """Start the daemon (blocking)."""
        try:
            import evdev  # noqa: F401
        except ImportError:
            verr(
                "[hotkeyd] 'evdev' package not installed. "
                "Install with: pip install evdev"
            )
            return

        self._running = True
        mod_str = f"{self.trigger_key_modifier} + " if self.trigger_key_modifier else ""
        print(f"[hotkeyd] Listening for {self.mode} on {mod_str}{self.trigger_key} "
              f"(code={self.key_code})")
        if self.key_code_2 is not None:
            mod2_str = f"{self.trigger_key_2_modifier} + " if self.trigger_key_2_modifier else ""
            print(f"[hotkeyd] Second PTT key: {mod2_str}{self.trigger_key_2} "
                  f"(code={self.key_code_2}, lang={self.trigger_key_2_lang})")
        print(f"[hotkeyd] IPC target: {self.socket_path}")

        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            print("\n[hotkeyd] Stopped.")
        finally:
            self._running = False

    async def _run_async(self):
        """Main async loop — find keyboards and monitor."""
        import evdev

        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        # Filter for devices that have EV_KEY capability
        key_devices = []
        for dev in devices:
            caps = dev.capabilities(verbose=False)
            if 1 in caps:  # EV_KEY = 1
                key_devices.append(dev)
                verbo(f"[hotkeyd] Monitoring: {dev.name} ({dev.path})")

        if not key_devices:
            verr(
                "[hotkeyd] No keyboard devices found. "
                "Ensure you are in the 'input' group: "
                "sudo usermod -aG input $USER && re-login"
            )
            return

        # In PTT mode with suppress enabled, set up a UInput proxy so that
        # the trigger key is consumed while all other keys pass through.
        # Only grab devices that actually have the trigger keycode — avoids
        # grabbing touchpads, mice, and other devices that can never produce
        # the trigger key.
        # Collect all trigger key codes that need grab suppression
        _grab_codes = {self.key_code}
        if self.key_code_2 is not None:
            _grab_codes.add(self.key_code_2)

        if self.mode == "ptt" and self.suppress_original:
            tasks = []
            for dev in key_devices:
                caps = dev.capabilities(verbose=False)
                key_caps = caps.get(1, [])  # EV_KEY codes
                if _grab_codes & set(key_caps):
                    tasks.append(self._monitor_device_ptt_grab(dev))
                else:
                    verbo(f"[hotkeyd] Skipping grab for {dev.name} "
                          f"(no trigger keycodes)")
                    tasks.append(self._monitor_device(dev))
        else:
            tasks = [self._monitor_device(kb) for kb in key_devices]
        await asyncio.gather(*tasks, return_exceptions=True)

    def _update_modifier_state(self, event_code: int, event_value: int):
        """Track modifier key held state across all devices."""
        if event_value == 1:  # key down
            self._modifier_held.add(event_code)
        elif event_value == 0:  # key up
            self._modifier_held.discard(event_code)

    def _modifier_active(self, modifier_code: Optional[int]) -> bool:
        """Check if a modifier key is currently held (or no modifier required)."""
        if modifier_code is None:
            return True
        return modifier_code in self._modifier_held

    async def _monitor_device(self, device):
        """Monitor a single input device for the trigger pattern."""
        import evdev

        try:
            async for event in device.async_read_loop():
                if not self._running:
                    break
                if event.type != evdev.ecodes.EV_KEY:
                    continue

                # Track modifier state for all modifier keys we care about
                self._update_modifier_state(event.code, event.value)

                # Second trigger key (always PTT mode)
                if self.key_code_2 is not None and event.code == self.key_code_2:
                    if self._modifier_active(self.modifier_code_2):
                        self._handle_ptt_2(event.value)
                    continue

                if event.code != self.key_code:
                    continue

                if not self._modifier_active(self.modifier_code):
                    continue

                if self.mode == "double_tap":
                    self._handle_double_tap(event.value)
                elif self.mode == "hold":
                    self._handle_hold(event.value)
                elif self.mode == "single":
                    self._handle_single(event.value)
                elif self.mode == "ptt":
                    self._handle_ptt(event.value)
        except OSError:
            # Device disconnected
            verbo(f"[hotkeyd] Device disconnected: {device.name}")

    async def _monitor_device_ptt_grab(self, device):
        """Monitor with evdev grab — suppress trigger key, forward everything else.

        Grabs the device for exclusive access and creates a UInput virtual
        device that proxies all events *except* the trigger key.  This
        prevents the held PTT key from reaching the focused application
        (e.g. Page Down scrolling a text editor).
        """
        import evdev

        try:
            ui = evdev.UInput.from_device(device, name=f"voxd-proxy-{device.name}")
        except Exception as e:
            verr(f"[hotkeyd] Cannot create UInput proxy for {device.name}: {e} — "
                 "falling back to non-grab mode")
            await self._monitor_device(device)
            return

        try:
            device.grab()
            verbo(f"[hotkeyd] Grabbed {device.name} (PTT suppress)")
        except Exception as e:
            verr(f"[hotkeyd] Cannot grab {device.name}: {e} — "
                 "falling back to non-grab mode")
            ui.close()
            await self._monitor_device(device)
            return

        try:
            async for event in device.async_read_loop():
                if not self._running:
                    break
                if event.type == evdev.ecodes.EV_KEY:
                    # Track modifier state
                    self._update_modifier_state(event.code, event.value)

                    # Primary trigger key
                    if event.code == self.key_code:
                        if self._modifier_active(self.modifier_code):
                            # Modifier held: consume key, handle PTT
                            self._handle_ptt(event.value)
                            continue
                        # No modifier (or modifier not held): forward normally
                        # so PgDn/PgUp work as expected
                    # Second trigger key
                    elif self.key_code_2 is not None and event.code == self.key_code_2:
                        if self._modifier_active(self.modifier_code_2):
                            # Modifier held: consume key, handle PTT
                            self._handle_ptt_2(event.value)
                            continue
                        # No modifier: forward normally
                # Everything else: forward to virtual device as-is.
                # Do NOT inject extra SYN_REPORT — let the original
                # SYN events flow through to preserve multi-axis frame
                # integrity (critical for touchpads/mice).
                ui.write_event(event)
        except OSError:
            verbo(f"[hotkeyd] Device disconnected: {device.name}")
        finally:
            try:
                device.ungrab()
            except Exception:
                pass
            ui.close()

    def _handle_double_tap(self, value: int):
        """Detect double-tap pattern (two quick key-up events)."""
        if value == 0:  # key up
            now = time.monotonic()
            if now - self._last_tap_time < self.double_tap_window:
                self._fire_trigger()
                self._last_tap_time = 0.0  # reset to prevent triple-tap
            else:
                self._last_tap_time = now

    def _handle_hold(self, value: int):
        """Detect hold pattern (key held longer than threshold)."""
        if value == 1:  # key down
            self._hold_start = time.monotonic()
        elif value == 0:  # key up
            if self._hold_start > 0:
                held = time.monotonic() - self._hold_start
                if held >= self.hold_threshold:
                    self._fire_trigger()
                self._hold_start = 0.0

    def _handle_single(self, value: int):
        """Fire on every single key-up event."""
        if value == 0:  # key up
            self._fire_trigger()

    def _handle_ptt(self, value: int):
        """Push-to-talk: key-down starts recording, key-up stops recording."""
        if value == 1:  # key down
            self._send_ipc(b"start_record")
            verbo("[hotkeyd] PTT key down — start_record sent")
        elif value == 0:  # key up
            self._send_ipc(b"stop_record")
            verbo("[hotkeyd] PTT key up — stop_record sent")

    def _handle_ptt_2(self, value: int):
        """Push-to-talk for second key: sends lang-tagged IPC commands."""
        lang = self.trigger_key_2_lang
        if value == 1:  # key down
            self._send_ipc(f"start_record:{lang}".encode())
            verbo(f"[hotkeyd] PTT key2 down — start_record:{lang} sent")
        elif value == 0:  # key up
            self._send_ipc(f"stop_record:{lang}".encode())
            verbo(f"[hotkeyd] PTT key2 up — stop_record:{lang} sent")

    def _fire_trigger(self):
        """Send trigger_record to VOXD's IPC socket."""
        self._send_ipc(b"trigger_record")
        verbo("[hotkeyd] Trigger sent!")

    def _send_ipc(self, command: bytes):
        """Send a command to VOXD's IPC socket."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(self.socket_path)
            sock.sendall(command)
            sock.close()
        except ConnectionRefusedError:
            verbo("[hotkeyd] VOXD not running (connection refused)")
        except FileNotFoundError:
            verbo("[hotkeyd] VOXD socket not found — is VOXD running?")
        except Exception as e:
            verr(f"[hotkeyd] Failed to send command: {e}")

    def stop(self):
        """Signal the daemon to stop."""
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_hotkey_daemon(cfg=None):
    """Start the hotkey daemon using config values (or defaults)."""
    trigger_key = "KEY_CAPSLOCK"
    trigger_key_modifier = ""
    trigger_key_2 = ""
    trigger_key_2_modifier = ""
    trigger_key_2_lang = "hu"
    mode = "double_tap"
    double_tap_window_ms = 350
    hold_threshold_ms = 300
    suppress_original = True

    if cfg is not None:
        trigger_key = cfg.data.get("hotkey_trigger_key", trigger_key)
        trigger_key_modifier = cfg.data.get("hotkey_trigger_key_modifier", trigger_key_modifier)
        trigger_key_2 = cfg.data.get("hotkey_trigger_key_2", trigger_key_2)
        trigger_key_2_modifier = cfg.data.get("hotkey_trigger_key_2_modifier", trigger_key_2_modifier)
        trigger_key_2_lang = cfg.data.get("hotkey_trigger_key_2_lang", trigger_key_2_lang)
        mode = cfg.data.get("hotkey_mode", mode)
        double_tap_window_ms = int(cfg.data.get("hotkey_double_tap_window_ms", double_tap_window_ms))
        hold_threshold_ms = int(cfg.data.get("hotkey_hold_threshold_ms", hold_threshold_ms))
        suppress_original = bool(cfg.data.get("hotkey_suppress_original", suppress_original))

    daemon = HotkeyDaemon(
        trigger_key=trigger_key,
        trigger_key_modifier=trigger_key_modifier,
        trigger_key_2=trigger_key_2,
        trigger_key_2_modifier=trigger_key_2_modifier,
        trigger_key_2_lang=trigger_key_2_lang,
        mode=mode,
        double_tap_window_ms=double_tap_window_ms,
        hold_threshold_ms=hold_threshold_ms,
        suppress_original=suppress_original,
    )
    daemon.run()
