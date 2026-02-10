"""Silero VAD wrapper for hybrid speech detection in Flux mode.

Provides a neural VAD that confirms speech presence after the
energy-based FluxVAD passes the initial gate.  This eliminates
false triggers from keyboard clicks, AC hum, and other
non-speech transients.

Supports two backends:
  - ONNX Runtime (preferred, ~20MB, fast CPU inference)
  - PyTorch (fallback, ~150MB+)

Install one of:
  pip install onnxruntime   # lightweight
  pip install torch         # heavier but more flexible
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from voxd.utils.libw import verbo, verr


# ---------------------------------------------------------------------------
# Cache dir for downloaded model files
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "voxd-plus" / "silero-vad"


class SileroVAD:
    """Thin wrapper around Silero VAD v5 for per-frame speech classification."""

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        backend: Optional[str] = None,
    ):
        """
        Args:
            threshold: Confidence above which a frame is classified as speech.
            sample_rate: Audio sample rate (must be 8000 or 16000 for Silero).
            backend: "onnx" or "torch". If None, auto-detect.
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._backend = backend or self._detect_backend()
        self._model = None
        self._initialized = False

    def _detect_backend(self) -> str:
        """Pick the best available backend."""
        try:
            import onnxruntime  # noqa: F401
            return "onnx"
        except ImportError:
            pass
        try:
            import torch  # noqa: F401
            return "torch"
        except ImportError:
            pass
        return "none"

    def initialize(self) -> bool:
        """Load the model. Returns True on success."""
        if self._initialized:
            return True

        if self._backend == "onnx":
            return self._init_onnx()
        elif self._backend == "torch":
            return self._init_torch()
        else:
            verr(
                "[silero_vad] Neither onnxruntime nor torch is installed. "
                "Install one: pip install onnxruntime"
            )
            return False

    def _init_onnx(self) -> bool:
        """Initialize using ONNX Runtime."""
        try:
            import onnxruntime as ort

            model_path = self._ensure_onnx_model()
            if model_path is None:
                return False

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._model = ort.InferenceSession(
                str(model_path), sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            # Initialize hidden state (Silero VAD v5 uses h/c LSTM states)
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
            self._initialized = True
            verbo("[silero_vad] Initialized with ONNX backend")
            return True
        except Exception as e:
            verr(f"[silero_vad] ONNX init failed: {e}")
            return False

    def _init_torch(self) -> bool:
        """Initialize using PyTorch (torch.hub)."""
        try:
            import torch

            model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                trust_repo=True, onnx=False,
            )
            model.eval()
            self._model = model
            self._initialized = True
            verbo("[silero_vad] Initialized with PyTorch backend")
            return True
        except Exception as e:
            verr(f"[silero_vad] PyTorch init failed: {e}")
            return False

    def _ensure_onnx_model(self) -> Optional[Path]:
        """Download the Silero VAD ONNX model if not cached."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = _CACHE_DIR / "silero_vad.onnx"

        if model_path.exists():
            return model_path

        url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
        verbo(f"[silero_vad] Downloading model from {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(model_path))
            verbo(f"[silero_vad] Model saved to {model_path}")
            return model_path
        except Exception as e:
            verr(f"[silero_vad] Download failed: {e}")
            return None

    def is_speech(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """Classify a single audio frame.

        Args:
            audio_frame: 1D float32 array at self.sample_rate.

        Returns:
            (is_speech, confidence) where confidence is 0.0–1.0.
        """
        if not self._initialized:
            if not self.initialize():
                return False, 0.0

        if self._backend == "onnx":
            return self._infer_onnx(audio_frame)
        elif self._backend == "torch":
            return self._infer_torch(audio_frame)
        return False, 0.0

    def _infer_onnx(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Run inference with ONNX Runtime."""
        try:
            # Silero VAD expects chunks of specific sizes
            # For 16kHz: 512 samples (32ms) — we may need to pad/truncate
            frame = frame.astype(np.float32).flatten()
            expected_size = 512 if self.sample_rate == 16000 else 256
            if len(frame) < expected_size:
                frame = np.pad(frame, (0, expected_size - len(frame)))
            elif len(frame) > expected_size:
                # Process in the expected chunk size, use last chunk
                frame = frame[-expected_size:]

            input_data = frame.reshape(1, -1)
            sr = np.array([self.sample_rate], dtype=np.int64)

            ort_inputs = {
                "input": input_data,
                "sr": sr,
                "h": self._h,
                "c": self._c,
            }
            output, h_new, c_new = self._model.run(None, ort_inputs)
            self._h = h_new
            self._c = c_new

            confidence = float(output.flatten()[0])
            return confidence > self.threshold, confidence
        except Exception as e:
            verr(f"[silero_vad] ONNX inference error: {e}")
            return False, 0.0

    def _infer_torch(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Run inference with PyTorch."""
        try:
            import torch

            tensor = torch.from_numpy(frame.astype(np.float32).flatten())
            confidence = self._model(tensor, self.sample_rate).item()
            return confidence > self.threshold, float(confidence)
        except Exception as e:
            verr(f"[silero_vad] PyTorch inference error: {e}")
            return False, 0.0

    def reset(self):
        """Reset internal LSTM states (call between utterances)."""
        if self._backend == "onnx":
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        elif self._backend == "torch" and self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass
