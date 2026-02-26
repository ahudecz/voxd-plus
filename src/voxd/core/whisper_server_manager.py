"""
WhisperServerManager: Persistent whisper-server for fast transcription.

Keeps the whisper model loaded in RAM so each transcription request
avoids the ~2s model-loading overhead of spawning whisper-cli.
"""

import subprocess
import time
import atexit
import signal
import os
from pathlib import Path
from typing import Optional
from voxd.utils.libw import verbo, verr


class WhisperServerManager:
    """Manages whisper-server lifecycle for transcription."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._port = 8178
        self._host = "127.0.0.1"
        self._url = f"http://{self._host}:{self._port}"
        self._startup_timeout = 30
        self._shutdown_timeout = 10
        self._model_path: Optional[str] = None

        atexit.register(self.stop_server)

    def is_server_running(self) -> bool:
        """Check if whisper-server is responding to health checks."""
        try:
            import requests
            response = requests.get(f"{self._url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    @property
    def model_path(self) -> Optional[str]:
        """The model currently loaded by the server."""
        return self._model_path

    def start_server(self, server_path: str, model_path: str,
                     port: int = 8178, host: str = "127.0.0.1",
                     language: str = "en", threads: int = 0,
                     beam_size: int = 5) -> bool:
        """Start whisper-server if not already running with the same model.

        Returns True if server is running, False on failure.
        """
        self._port = port
        self._host = host
        self._url = f"http://{host}:{port}"

        # If already running with the same model, reuse
        if self._model_path == model_path and self.is_server_running():
            verbo(f"[whisper-server] Already running on {self._url} with {Path(model_path).name}")
            return True

        # If running with a different model, stop first
        if self._process is not None:
            verbo("[whisper-server] Stopping to load different model")
            self.stop_server()

        if not Path(server_path).exists():
            verr(f"[whisper-server] Binary not found: {server_path}")
            return False
        if not Path(model_path).exists():
            verr(f"[whisper-server] Model not found: {model_path}")
            return False

        verbo(f"[whisper-server] Starting on {self._url}")
        verbo(f"[whisper-server] Model: {Path(model_path).name}")

        # Determine thread count
        if threads <= 0:
            cpu_count = os.cpu_count() or 4
            threads = min(12, max(4, cpu_count // 2))

        cmd = [
            server_path,
            "--model", model_path,
            "--port", str(port),
            "--host", host,
            "-t", str(threads),
            "-l", language,
        ]

        # No --prompt here: prompt is passed per-request so each language
        # gets the correct vocabulary hints.

        if beam_size > 0 and beam_size != 5:
            cmd.extend(["-bs", str(beam_size)])

        verbo(f"[whisper-server] Command: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )

            # Pin whisper-server to the first half of CPU cores so it
            # doesn't compete with other heavy processes (e.g. Claude Code).
            try:
                cpu_count = os.cpu_count() or 4
                half = max(2, cpu_count // 2)
                os.sched_setaffinity(self._process.pid, set(range(half)))
                verbo(f"[whisper-server] Pinned to cores 0-{half - 1}")
            except (OSError, AttributeError):
                pass  # sched_setaffinity not available on all platforms

            start_time = time.time()
            while time.time() - start_time < self._startup_timeout:
                if self.is_server_running():
                    self._model_path = model_path
                    verbo(f"[whisper-server] Ready on {self._url} "
                          f"(startup: {time.time() - start_time:.1f}s)")
                    return True

                if self._process.poll() is not None:
                    verr(f"[whisper-server] Process exited (code: {self._process.returncode})")
                    self._process = None
                    return False

                time.sleep(0.5)

            verr("[whisper-server] Timeout waiting for server to start")
            self.stop_server()
            return False

        except Exception as e:
            verr(f"[whisper-server] Failed to start: {e}")
            self._process = None
            return False

    def stop_server(self):
        """Stop the whisper-server process gracefully."""
        if self._process is None:
            return

        verbo("[whisper-server] Stopping server...")

        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            try:
                self._process.wait(timeout=self._shutdown_timeout)
                verbo("[whisper-server] Stopped gracefully")
            except subprocess.TimeoutExpired:
                verbo("[whisper-server] Force killing...")
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
                verbo("[whisper-server] Force killed")
        except (ProcessLookupError, OSError):
            pass
        except Exception as e:
            verr(f"[whisper-server] Error during shutdown: {e}")
        finally:
            self._process = None
            self._model_path = None

    def is_process_alive(self) -> bool:
        """Check if the server process is still running (no HTTP request)."""
        return self._process is not None and self._process.poll() is None

    def transcribe(self, audio_path: str, language: str = "",
                   prompt: str = "",
                   response_format: str = "text") -> Optional[str]:
        """Transcribe audio via whisper-server HTTP API.

        Args:
            audio_path: Path to WAV audio file
            language: Language code (e.g. "en", "hu"). Empty = use server default.
            prompt: Whisper vocabulary hint prompt. Overrides server default.
            response_format: "text" or "json"

        Returns:
            Transcribed text, or None on failure.
        """
        if not self.is_process_alive():
            return None

        try:
            import requests

            with open(audio_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f, "audio/wav")}
                data = {"response_format": response_format}
                if language:
                    data["language"] = language
                if prompt:
                    data["prompt"] = prompt

                response = requests.post(
                    f"{self._url}/inference",
                    files=files,
                    data=data,
                    timeout=30,
                )

            if response.status_code != 200:
                verr(f"[whisper-server] HTTP {response.status_code}: {response.text[:200]}")
                return None

            if response_format == "json":
                return response.json().get("text", "").strip()
            else:
                return response.text.strip()

        except Exception as e:
            verr(f"[whisper-server] Transcription request failed: {e}")
            return None

    def get_server_url(self) -> str:
        return self._url

    def get_server_status(self) -> dict:
        return {
            "process_running": self._process is not None and self._process.poll() is None,
            "server_responding": self.is_server_running(),
            "url": self._url,
            "model": Path(self._model_path).name if self._model_path else None,
            "pid": self._process.pid if self._process else None,
        }


# Global instance
_manager = WhisperServerManager()


def get_whisper_server_manager() -> WhisperServerManager:
    """Get the global whisper-server manager instance."""
    return _manager


def ensure_whisper_server_running(cfg) -> bool:
    """Start whisper-server using the app config.

    This is the main entry point for GUI/tray startup.
    """
    try:
        from voxd.paths import _locate_whisper_server
        server_path = str(_locate_whisper_server())
    except FileNotFoundError:
        verbo("[whisper-server] Binary not found, transcription will use whisper-cli subprocess")
        return False

    # Use the multilingual model so the server handles all languages.
    # If the config points to an English-only (.en.bin) model, try the
    # multilingual sibling first so both EN and HU work via one server.
    model_path = cfg.data.get("whisper_model_path", "")
    if model_path and model_path.endswith(".en.bin"):
        multi_path = model_path.replace(".en.bin", ".bin")
        if Path(multi_path).exists():
            model_path = multi_path
            verbo(f"[whisper-server] Using multilingual model: {Path(model_path).name}")
    if not model_path or not Path(model_path).exists():
        verbo("[whisper-server] Model not found, skipping server start")
        return False

    port = cfg.data.get("whisper_server_port", 8178)
    language = cfg.data.get("language", "en")
    beam_size = int(cfg.data.get("whisper_beam_size", 5))

    return _manager.start_server(
        server_path=server_path,
        model_path=model_path,
        port=port,
        language=language,
        beam_size=beam_size,
    )
