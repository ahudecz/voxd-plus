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
    "KEY_RIGHTCTRL": 97,
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
        mode: str = "double_tap",
        double_tap_window_ms: int = 350,
        hold_threshold_ms: int = 300,
        suppress_original: bool = True,
        socket_path: Optional[str] = None,
    ):
        self.trigger_key = trigger_key
        self.key_code = _resolve_key_code(trigger_key)
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
        print(f"[hotkeyd] Listening for {self.mode} on {self.trigger_key} "
              f"(code={self.key_code})")
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
        if self.mode == "ptt" and self.suppress_original:
            tasks = []
            for dev in key_devices:
                caps = dev.capabilities(verbose=False)
                key_caps = caps.get(1, [])  # EV_KEY codes
                if self.key_code in key_caps:
                    tasks.append(self._monitor_device_ptt_grab(dev))
                else:
                    verbo(f"[hotkeyd] Skipping grab for {dev.name} "
                          f"(no keycode {self.key_code})")
                    tasks.append(self._monitor_device(dev))
        else:
            tasks = [self._monitor_device(kb) for kb in key_devices]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _monitor_device(self, device):
        """Monitor a single input device for the trigger pattern."""
        import evdev

        try:
            async for event in device.async_read_loop():
                if not self._running:
                    break
                if event.type != evdev.ecodes.EV_KEY:
                    continue
                if event.code != self.key_code:
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
                # Trigger key: consume (don't forward), handle PTT
                if event.type == evdev.ecodes.EV_KEY and event.code == self.key_code:
                    self._handle_ptt(event.value)
                    continue
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
    mode = "double_tap"
    double_tap_window_ms = 350
    hold_threshold_ms = 300
    suppress_original = True

    if cfg is not None:
        trigger_key = cfg.data.get("hotkey_trigger_key", trigger_key)
        mode = cfg.data.get("hotkey_mode", mode)
        double_tap_window_ms = int(cfg.data.get("hotkey_double_tap_window_ms", double_tap_window_ms))
        hold_threshold_ms = int(cfg.data.get("hotkey_hold_threshold_ms", hold_threshold_ms))
        suppress_original = bool(cfg.data.get("hotkey_suppress_original", suppress_original))

    daemon = HotkeyDaemon(
        trigger_key=trigger_key,
        mode=mode,
        double_tap_window_ms=double_tap_window_ms,
        hold_threshold_ms=hold_threshold_ms,
        suppress_original=suppress_original,
    )
    daemon.run()
