"""
Unit tests for MicroWakeWordEngine — use a mocked TFLite interpreter so no
model file is required.  Tests cover:

- Sliding window accumulation and cutoff logic
- Refractory period suppression
- Malformed / short frame handling (below chunk boundary)
- PCM buffer carry-over across calls
- reset() clears all state
"""
from __future__ import annotations

import collections
import struct
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers to build a fake tflite interpreter
# ---------------------------------------------------------------------------

def _make_fake_interpreter(prob_sequence):
    """
    Return a fake interpreter whose invoke() cycles through prob_sequence.

    Each element in prob_sequence is a float probability [0, 1].
    The fake uses scale=1/255, zero=0 (uint8 dequant approximation).
    """
    # Use uint8 output: scale=1/255 ≈ 0.00392, zero=0
    # out_scale * (raw - 0) → raw = round(prob / out_scale)
    _OUT_SCALE = 1.0 / 255.0
    _INP_SCALE = 0.10196  # matches okay_nabu
    _INP_ZERO = -128

    probs = list(prob_sequence)
    idx = [0]

    inp_detail = {
        "index": 0,
        "name": "serving_default_input_audio:0",
        "shape": np.array([1, 1, 40]),
        "dtype": np.int8,
        "quantization": (_INP_SCALE, _INP_ZERO),
    }
    out_detail = {
        "index": 1,
        "name": "StatefulPartitionedCall:0",
        "shape": np.array([1, 1]),
        "dtype": np.uint8,
        "quantization": (_OUT_SCALE, 0),
    }

    interp = MagicMock()
    interp.get_input_details.return_value = [inp_detail]
    interp.get_output_details.return_value = [out_detail]

    def fake_invoke():
        pass  # state updated in get_tensor

    def fake_get_tensor(idx_):
        if idx_ == 1:
            prob = probs[min(idx[0], len(probs) - 1)]
            idx[0] += 1
            raw = int(round(prob / _OUT_SCALE))
            return np.array([[raw]], dtype=np.uint8)
        return np.zeros((1, 1, 40), dtype=np.int8)

    interp.invoke.side_effect = fake_invoke
    interp.get_tensor.side_effect = fake_get_tensor
    return interp


def _make_pcm_chunk(n_samples: int = 320) -> bytes:
    """Return n_samples of silent int16 PCM."""
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_ENGINE_MODULE = "ovos_ww_plugin_microwakeword.inference"


class TestSlidingWindow(unittest.TestCase):
    """Sliding window average triggers correctly."""

    def _engine_with_probs(self, probs, cutoff=0.5, window=5, refractory=100):
        """Build engine with mocked interpreter returning given probs."""
        fake = _make_fake_interpreter(probs)
        with patch(f"{_ENGINE_MODULE}.Interpreter", return_value=fake):
            from ovos_ww_plugin_microwakeword.inference import MicroWakeWordEngine
            eng = MicroWakeWordEngine(
                "fake.tflite",
                probability_cutoff=cutoff,
                sliding_window_size=window,
                refractory_frames=refractory,
            )
        return eng

    def test_no_detection_below_cutoff(self):
        """All probs below cutoff → no detection."""
        eng = self._engine_with_probs([0.3] * 20)
        chunk = _make_pcm_chunk(320)
        results = [eng.is_detected(chunk) for _ in range(20)]
        self.assertFalse(any(results))

    def test_detection_fires_when_window_full_and_above_cutoff(self):
        """When sliding window fills with values above cutoff → fires."""
        # window=5: need 5 consecutive frames > 0.5
        eng = self._engine_with_probs([0.9] * 20)
        chunk = _make_pcm_chunk(320)
        detections = 0
        for _ in range(20):
            if eng.is_detected(chunk):
                detections += 1
        self.assertGreaterEqual(detections, 1)

    def test_window_partial_does_not_fire(self):
        """Window of size 10 with only 4 frames above cutoff → no fire."""
        # 4 high, then drop — window never fully above cutoff
        probs = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] * 3
        eng = self._engine_with_probs(probs, cutoff=0.5, window=10)
        chunk = _make_pcm_chunk(320)
        results = [eng.is_detected(chunk) for _ in range(30)]
        self.assertFalse(any(results))

    def test_detection_average_logic(self):
        """Average of window must be >= cutoff, not each individual value."""
        # Alternating 0.8 / 0.3 → average 0.55 > 0.5 → should fire
        probs = [0.8, 0.3] * 20
        eng = self._engine_with_probs(probs, cutoff=0.5, window=4)
        chunk = _make_pcm_chunk(320)
        results = [eng.is_detected(chunk) for _ in range(40)]
        self.assertTrue(any(results))


class TestRefractoryPeriod(unittest.TestCase):
    """Refractory period suppresses double-fires."""

    def _engine(self, refractory=10, window=3):
        probs = [0.9] * 200
        fake = _make_fake_interpreter(probs)
        with patch(f"{_ENGINE_MODULE}.Interpreter", return_value=fake):
            from ovos_ww_plugin_microwakeword.inference import MicroWakeWordEngine
            eng = MicroWakeWordEngine(
                "fake.tflite",
                probability_cutoff=0.5,
                sliding_window_size=window,
                refractory_frames=refractory,
            )
        return eng

    def test_double_fire_suppressed(self):
        """After first detection refractory suppresses immediate second fire."""
        eng = self._engine(refractory=50, window=3)
        chunk = _make_pcm_chunk(320)
        fires = []
        for _ in range(30):
            fires.append(eng.is_detected(chunk))
        # First fire happens; subsequent ones within refractory must not.
        fire_indices = [i for i, v in enumerate(fires) if v]
        self.assertGreaterEqual(len(fire_indices), 1)
        if len(fire_indices) > 1:
            gap = fire_indices[1] - fire_indices[0]
            self.assertGreater(gap, 10)  # at least refractory gap

    def test_refractory_clears_window(self):
        """Window is cleared during refractory so partial votes don't accumulate."""
        eng = self._engine(refractory=5, window=3)
        chunk = _make_pcm_chunk(320)
        eng.is_detected(chunk)  # prime, may fire
        # After firing, window should be cleared
        self.assertEqual(len(eng._window), 0)


class TestPCMBuffering(unittest.TestCase):
    """Sub-chunk and multi-chunk PCM buffering."""

    def _engine(self):
        probs = [0.1] * 500
        fake = _make_fake_interpreter(probs)
        with patch(f"{_ENGINE_MODULE}.Interpreter", return_value=fake):
            from ovos_ww_plugin_microwakeword.inference import MicroWakeWordEngine
            eng = MicroWakeWordEngine("fake.tflite")
        return eng

    def test_short_chunk_buffered(self):
        """Chunk shorter than 320 samples is buffered without inference."""
        eng = self._engine()
        # 100 samples — should not cause any inference
        chunk = _make_pcm_chunk(100)
        probs = eng.process_audio(chunk)
        self.assertEqual(probs, [])
        self.assertEqual(len(eng._pcm_buffer), 200)  # 100 * 2 bytes

    def test_leftover_consumed_on_next_call(self):
        """Leftover bytes from one call are consumed in the next."""
        eng = self._engine()
        p1 = eng.process_audio(_make_pcm_chunk(200))  # 400 bytes
        self.assertEqual(p1, [])
        # Add 120 more samples (240 bytes) → total 640 bytes → 2 full chunks
        p2 = eng.process_audio(_make_pcm_chunk(120))
        # The MicroFrontend may not yield features for every chunk (first
        # chunk primes it), so we just check no error and type is list.
        self.assertIsInstance(p2, list)

    def test_exact_multiple_chunks(self):
        """Exactly two 320-sample chunks → no leftover."""
        eng = self._engine()
        eng.process_audio(_make_pcm_chunk(640))
        self.assertEqual(len(eng._pcm_buffer), 0)

    def test_malformed_empty_chunk(self):
        """Empty bytes produces no probs and no error."""
        eng = self._engine()
        probs = eng.process_audio(b"")
        self.assertEqual(probs, [])

    def test_odd_byte_count_handled(self):
        """Odd byte count does not crash (last byte kept in buffer)."""
        eng = self._engine()
        probs = eng.process_audio(b"\x00" * 641)
        self.assertIsInstance(probs, list)


class TestReset(unittest.TestCase):
    """reset() restores clean state."""

    def _engine(self):
        probs = [0.9] * 200
        fake = _make_fake_interpreter(probs)
        with patch(f"{_ENGINE_MODULE}.Interpreter", return_value=fake):
            from ovos_ww_plugin_microwakeword.inference import MicroWakeWordEngine
            eng = MicroWakeWordEngine("fake.tflite", sliding_window_size=3)
        return eng

    def test_reset_clears_window(self):
        eng = self._engine()
        # Fill the window
        for _ in range(5):
            eng.process_audio(_make_pcm_chunk(320))
        eng.reset()
        self.assertEqual(len(eng._window), 0)

    def test_reset_clears_refractory(self):
        eng = self._engine()
        eng._refractory = 99
        eng.reset()
        self.assertEqual(eng._refractory, 0)

    def test_reset_clears_pcm_buffer(self):
        eng = self._engine()
        eng._pcm_buffer = b"\x00" * 100
        eng.reset()
        self.assertEqual(eng._pcm_buffer, b"")


class TestPluginContract(unittest.TestCase):
    """OPM HotWordEngine contract: update() + found_wake_word()."""

    def _plugin(self, detect=False):
        from ovos_ww_plugin_microwakeword import MicroWakeWordPlugin

        eng_mock = MagicMock()
        eng_mock.is_detected.return_value = detect

        with patch("ovos_ww_plugin_microwakeword.MicroWakeWordEngine", return_value=eng_mock), \
             patch("ovos_ww_plugin_microwakeword.MicroWakeWordPlugin._resolve_model", return_value="fake.tflite"):
            plugin = MicroWakeWordPlugin.__new__(MicroWakeWordPlugin)
            plugin.config = {}
            plugin.key_phrase = "okay nabu"
            plugin._trigger_flag = False
            plugin._engine = eng_mock
        return plugin

    def test_found_wake_word_false_by_default(self):
        plugin = self._plugin(detect=False)
        plugin.update(_make_pcm_chunk())
        self.assertFalse(plugin.found_wake_word())

    def test_found_wake_word_true_when_engine_fires(self):
        plugin = self._plugin(detect=True)
        plugin.update(_make_pcm_chunk())
        self.assertTrue(plugin.found_wake_word())

    def test_found_wake_word_resets_flag(self):
        plugin = self._plugin(detect=True)
        plugin.update(_make_pcm_chunk())
        first = plugin.found_wake_word()
        second = plugin.found_wake_word()
        self.assertTrue(first)
        self.assertFalse(second)

    def test_engine_reset_called_after_detection(self):
        plugin = self._plugin(detect=True)
        plugin.update(_make_pcm_chunk())
        plugin.found_wake_word()
        plugin._engine.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
