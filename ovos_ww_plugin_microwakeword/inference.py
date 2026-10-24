"""
microWakeWord TFLite inference engine.

Model input:  [1, stride, 40] int8 — ``stride`` 40-dim quantized log-mel
              feature slices from the TFLite Micro audio frontend. v1 models
              take one slice per inference with a 20 ms feature step; v2
              models take three slices with a 10 ms feature step, as their
              ESPHome manifests state (``feature_step_size``). The frontend
              always steps 10 ms; for v1 the second 10 ms of each 20 ms chunk
              is not read, which approximates ESPHome's 20 ms step.
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

    The model takes ``stride`` 40-dim quantized log-mel feature slices per
    inference, as its input shape states. Audio must be 16 kHz 16-bit mono
    PCM. Feed raw int16 bytes via :meth:`process_audio`; each call produces
    zero or more probability values, one per inference.

    Parameters
    ----------
    model_path:
        Path to a ``.tflite`` microWakeWord model file.
    probability_cutoff:
        Raw dequantized probability threshold in [0, 1].  A frame is
        considered positive when the model output exceeds this value.
        Default ``0.5``.
    sliding_window_size:
        Number of consecutive inferences whose average probability must
        reach ``probability_cutoff`` before a detection event is reported.
        Mirrors the ESPHome manifest's ``sliding_window_size`` (v2) or
        ``sliding_window_average_size`` (v1).  Default ``10``.
    refractory_frames:
        After a detection, ignore this many inferences before arming again.
        Default ``40`` (800 ms for a v1 model, 1.2 s for a v2 model).
    """

    SAMPLE_RATE = 16000
    # MicroFrontend reads 160 samples (10 ms) per call and emits one
    # 40-feature slice. A chunk of one feature step yields exactly one slice;
    # with a 20 ms step the second half of each chunk is not read.
    SAMPLES_PER_FEATURE = 160

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

        shape = [int(n) for n in inp["shape"]]
        if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != 40:
            raise ValueError(
                f"Unexpected model input shape {inp['shape']}; expected [1, stride, 40]."
            )
        self.stride = shape[1]
        step_ms = 20 if self.stride == 1 else 10
        self.input_chunk_samples = step_ms * self.SAMPLE_RATE // 1000
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
        # Feature slices waiting for the next inference
        self._features: List[List[float]] = []

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
            accepted; bytes that don't form a complete feature step are
            buffered internally and consumed on the next call.

        Returns
        -------
        List[float]
            One probability value (in [0, 1]) per inference.  Empty list when
            the input does not complete an inference.
        """
        self._pcm_buffer += pcm_bytes
        chunk_bytes = self.input_chunk_samples * 2  # int16 = 2 bytes/sample
        probs: List[float] = []

        while len(self._pcm_buffer) >= chunk_bytes:
            chunk = self._pcm_buffer[:chunk_bytes]
            self._pcm_buffer = self._pcm_buffer[chunk_bytes:]
            out = self._frontend.process_samples(chunk)
            if not out.features:
                continue
            self._features.append(out.features)
            if len(self._features) == self.stride:
                probs.append(self._infer(self._features))
                self._features = []

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
        """Reset all state: frontend, sliding window, refractory counter, buffers."""
        self._frontend.reset()
        self._window.clear()
        self._refractory = 0
        self._pcm_buffer = b""
        self._features = []
        # Re-allocate tensors to reset model's internal streaming state
        self._interp.allocate_tensors()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(self, features: List[List[float]]) -> float:
        """Quantize ``stride`` feature slices, run one inference step, dequantize output."""
        feat = np.array(features, dtype=np.float32)
        feat_q = np.clip(
            np.round(feat / self._inp_scale + self._inp_zero), -128, 127
        ).astype(np.int8)
        self._interp.set_tensor(self._inp_index, feat_q.reshape(1, self.stride, 40))
        self._interp.invoke()
        raw = self._interp.get_tensor(self._out_index)
        return float(self._out_scale * (int(raw[0][0]) - self._out_zero))
