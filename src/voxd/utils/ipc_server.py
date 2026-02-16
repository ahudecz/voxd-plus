import socket
import threading
from pathlib import Path

def _socket_path():
    return Path.home() / ".config" / "voxd-plus" / "voxd-plus.sock"

def start_ipc_server(trigger_callback, start_callback=None, stop_callback=None):
    """Starts a background thread that listens for IPC commands.

    Supported commands:
        trigger_record  — toggle recording (original behaviour)
        start_record    — start recording only (PTT key-down)
        stop_record     — stop recording only (PTT key-up)

    Parameters
    ----------
    trigger_callback : callable
        Called on ``trigger_record`` (toggle).
    start_callback : callable | None
        Called on ``start_record``.  Falls back to *trigger_callback* when None.
    stop_callback : callable | None
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
            conn, _ = server.accept()
            data = conn.recv(1024).strip()
            if data == b"trigger_record":
                trigger_callback()
            elif data == b"start_record":
                _start()
            elif data == b"stop_record":
                _stop()
            conn.close()

    t = threading.Thread(target=_serve_loop, daemon=True)
    t.start()
