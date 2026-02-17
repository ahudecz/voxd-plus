import socket
import threading
from pathlib import Path

def _socket_path():
    return Path.home() / ".config" / "voxd-plus" / "voxd-plus.sock"

def start_ipc_server(trigger_callback, start_callback=None, stop_callback=None):
    """Starts a background thread that listens for IPC commands.

    Supported commands:
        trigger_record          — toggle recording (original behaviour)
        start_record            — start recording only (PTT key-down)
        stop_record             — stop recording only (PTT key-up)
        start_record:<prompt>   — start recording with prompt override
        stop_record:<prompt>    — stop recording with prompt override

    Parameters
    ----------
    trigger_callback : callable
        Called on ``trigger_record`` (toggle).
    start_callback : callable(prompt_key=None) | None
        Called on ``start_record``.  Falls back to *trigger_callback* when None.
        Receives an optional *prompt_key* keyword argument when a prompt-tagged
        command is received (e.g. ``start_record:prompt1``).
    stop_callback : callable(prompt_key=None) | None
        Called on ``stop_record``.  Falls back to *trigger_callback* when None.
    """
    sock_path = _socket_path()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen()

    _start = start_callback or trigger_callback
    _stop = stop_callback or trigger_callback

    def _serve_loop():
        while True:
            try:
                conn, _ = server.accept()
                data = conn.recv(1024).strip().decode("utf-8", errors="replace")
                if data == "trigger_record":
                    trigger_callback()
                elif data.startswith("start_record"):
                    prompt_key = None
                    if ":" in data:
                        prompt_key = data.split(":", 1)[1] or None
                    try:
                        _start(prompt_key=prompt_key)
                    except TypeError:
                        _start()
                elif data.startswith("stop_record"):
                    prompt_key = None
                    if ":" in data:
                        prompt_key = data.split(":", 1)[1] or None
                    try:
                        _stop(prompt_key=prompt_key)
                    except TypeError:
                        _stop()
                conn.close()
            except Exception:
                pass

    t = threading.Thread(target=_serve_loop, daemon=True)
    t.start()
