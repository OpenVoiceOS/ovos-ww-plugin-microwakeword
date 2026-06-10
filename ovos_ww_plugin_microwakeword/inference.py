"""
microWakeWord TFLite inference engine.

Model input:  [1, 1, 40] int8  — one 40-dim quantized log-mel feature slice
              produced by the TFLite Micro audio frontend every 10 ms
              (160 samples at 16 kHz, 320-sample / 20 ms input stride).
Model output: [1, 1] uint8 — quantized probability (dequantize to [0, 1]).

State is baked into the model graph (streaming layers with resource variables),
so each sequential call to invoke() accumulates the internal RNN/conv state
automatically — no external state management needed.

Feature extraction uses pymicro-features which wraps the TFLite Micro audio
frontend (same frontend used by ESPHome / the original microWakeWord pipeline).
"""
from __future__ import annotations

import collections
from typing import List, Optional

import numpy as np

try:
    import ai_edge_litert.interpreter as _tflite_mod
    Interpreter = _tflite_mod.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as _tflite_mod
        Interpreter = _tflite_mod.Interpreter
    except ImportError:
        Interpreter = None  # surfaced as ImportError at plugin load time

from pymicro_features import MicroFrontend


class MicroWakeWordEngine:
    """
    Streaming inference wrapper around a microWakeWord TFLite model.

    The model expects one 40-dim quantized log-mel feature slice per call.
    Audio must be 16 kHz 16-bit mono PCM.  Feed raw int16 bytes via
    :meth:`process_audio`; each call produces zero or more probability
    values (one per 10 ms frame extracted from the audio).

    Parameters
    ----------
    model_path:
        Path to a ``.tflite`` microWakeWord model file.
    probability_cutoff:
        Raw dequantized probability threshold in [0, 1].  A frame is
        considered positive when the model output exceeds this value.
        Default ``0.5``.
    sliding_window_size:
        Number of consecutive frames that must exceed ``probability_cutoff``
        before a detection event is reported.  Mirrors the ESPHome
        ``sliding_window_average_size`` config key.  Default ``10``.
    refractory_frames:
        After a detection, ignore this many frames before arming again.
        Default ``40`` (≈ 400 ms at 10 ms/frame).
    """

    SAMPLE_RATE = 16000
    # MicroFrontend emits one 40-feature slice per 160 samples consumed.
    # Feed 320-sample (20 ms) chunks to get one feature slice per call.
    SAMPLES_PER_FEATURE = 160
    INPUT_CHUNK_SAMPLES = 320  # 20 ms; yields exactly one feature slice

    def __init__(
        self,
        model_path: str,
        probability_cutoff: float = 0.5,
        sliding_window_size: int = 10,
        refractory_frames: int = 40,
    ) -> None:
        if Interpreter is None:
            raise ImportError(
                "No TFLite runtime found. Install ai-edge-litert or tflite-runtime."
            )

        self.probability_cutoff = probability_cutoff
        self.sliding_window_size = sliding_window_size
        self.refractory_frames = refractory_frames

        self._frontend = MicroFrontend()
        self._interp = Interpreter(model_path)
        self._interp.allocate_tensors()

        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]

        if list(inp["shape"]) != [1, 1, 40]:
            raise ValueError(
                f"Unexpected model input shape {inp['shape']}; expected [1, 1, 40]."
            )
        if inp["dtype"] != np.int8:
            raise ValueError(
                f"Unexpected input dtype {inp['dtype']}; expected int8."
            )

        self._inp_index = inp["index"]
        self._out_index = out["index"]
        self._inp_scale: float = inp["quantization"][0]
        self._inp_zero: int = int(inp["quantization"][1])
        self._out_scale: float = out["quantization"][0]
        self._out_zero: int = int(out["quantization"][1])

        # Sliding window of recent per-frame probabilities
        self._window: collections.deque = collections.deque(
            maxlen=sliding_window_size
        )
        # Refractory counter (counts down after detection)
        self._refractory: int = 0
        # Leftover PCM bytes from the previous update call
        self._pcm_buffer: bytes = b""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_audio(self, pcm_bytes: bytes) -> List[float]:
        """
        Feed raw 16-bit PCM audio and return per-frame dequantized probabilities.

        Parameters
        ----------
        pcm_bytes:
            Raw int16 little-endian PCM bytes at 16 kHz.  Any length is
            accepted; bytes that don't form a complete 320-sample chunk are
            buffered internally and consumed on the next call.

        Returns
        -------
        List[float]
            One probability value (in [0, 1]) per 10 ms frame extracted from
            the input.  Empty list when the input is shorter than one chunk.
        """
        self._pcm_buffer += pcm_bytes
        chunk_bytes = self.INPUT_CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample
        probs: List[float] = []

        while len(self._pcm_buffer) >= chunk_bytes:
            chunk = self._pcm_buffer[:chunk_bytes]
            self._pcm_buffer = self._pcm_buffer[chunk_bytes:]
            out = self._frontend.process_samples(chunk)
            if not out.features:
                continue
            prob = self._infer(out.features)
            probs.append(prob)

        return probs

    def is_detected(self, pcm_bytes: bytes) -> bool:
        """
        Return True if a wake-word is detected in this audio chunk.

        Uses the sliding window average and refractory period logic that
        mirrors ESPHome's ``micro_wake_word`` component behaviour.

        Parameters
        ----------
        pcm_bytes:
            Raw int16 PCM bytes (16 kHz mono).

        Returns
        -------
        bool
            True exactly once per detection event; False otherwise.
        """
        probs = self.process_audio(pcm_bytes)
        for prob in probs:
            if self._refractory > 0:
                self._refractory -= 1
                self._window.clear()
                continue
            self._window.append(prob)
            if len(self._window) == self.sliding_window_size:
                avg = sum(self._window) / len(self._window)
                if avg >= self.probability_cutoff:
                    self._refractory = self.refractory_frames
                    self._window.clear()
                    return True
        return False

    def reset(self) -> None:
        """Reset all state: frontend, sliding window, refractory counter, PCM buffer."""
        self._frontend.reset()
        self._window.clear()
        self._refractory = 0
        self._pcm_buffer = b""
        # Re-allocate tensors to reset model's internal streaming state
        self._interp.allocate_tensors()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(self, features: List[float]) -> float:
        """Quantize features, run one inference step, dequantize output."""
        feat = np.array(features, dtype=np.float32)
        feat_q = np.clip(
            np.round(feat / self._inp_scale + self._inp_zero), -128, 127
        ).astype(np.int8)
        self._interp.set_tensor(self._inp_index, feat_q.reshape(1, 1, 40))
        self._interp.invoke()
        raw = self._interp.get_tensor(self._out_index)
        return float(self._out_scale * (int(raw[0][0]) - self._out_zero))
