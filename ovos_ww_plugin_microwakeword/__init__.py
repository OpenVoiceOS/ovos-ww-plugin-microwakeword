"""
ovos-ww-plugin-microwakeword — OVOS hotword plugin wrapping microWakeWord
TFLite streaming models from the ESPHome ecosystem.

Supported models: okay_nabu, hey_jarvis, alexa (v1 and v2), hey_mycroft (v2),
and any compatible community model published at
https://github.com/esphome/micro-wake-word-models.

Configuration keys (under the hotwords section in mycroft.conf):

    model (str):
        Absolute path to a local ``.tflite`` file, OR an https:// URL.
        Alternatively set ``model_name`` to auto-resolve from the official
        ESPHome model repository.  Defaults to ``okay_nabu``.

    model_name (str):
        Short name of an official ESPHome model (e.g. ``okay_nabu``,
        ``hey_jarvis``, ``alexa``).  Used when ``model`` is not given.
        Version suffix ``/v2`` is appended automatically when
        ``model_version`` is set to ``2``.

    model_version (int):
        ``1`` (default) or ``2``.  Selects the v1 or v2 model directory in
        the ESPHome repository.

    probability_cutoff (float):
        Dequantized probability threshold in [0, 1].  Default ``0.5``.

    sliding_window_size (int):
        Number of consecutive inferences whose average must reach
        ``probability_cutoff`` to trigger a detection.  Default ``10``.

    refractory_frames (int):
        Inferences to suppress after a detection (anti-double-fire).
        Default ``40`` (800 ms for a v1 model, 1.2 s for a v2 model).
"""
from __future__ import annotations

import hashlib
import os
from os.path import isfile, expanduser, join
from typing import Optional

import requests
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home

from ovos_ww_plugin_microwakeword.inference import MicroWakeWordEngine

_ESPHOME_BASE = (
    "https://github.com/esphome/micro-wake-word-models/raw/main/models"
)


def _model_url(name: str, version: int = 1) -> str:
    if version == 2:
        return f"{_ESPHOME_BASE}/v2/{name}.tflite"
    return f"{_ESPHOME_BASE}/{name}.tflite"


class MicroWakeWordPlugin(HotWordEngine):
    """OVOS HotWordEngine wrapping microWakeWord TFLite streaming models."""

    def __init__(self, key_phrase: str = "okay nabu", config: Optional[dict] = None) -> None:
        super().__init__(key_phrase, config)

        self._trigger_flag = False

        probability_cutoff = float(self.config.get("probability_cutoff", 0.5))
        sliding_window_size = int(self.config.get("sliding_window_size", 10))
        refractory_frames = int(self.config.get("refractory_frames", 40))

        model_path = self._resolve_model()

        LOG.info(f"Loading microWakeWord model: {model_path}")
        self._engine = MicroWakeWordEngine(
            model_path=model_path,
            probability_cutoff=probability_cutoff,
            sliding_window_size=sliding_window_size,
            refractory_frames=refractory_frames,
        )

    # ------------------------------------------------------------------
    # Model resolution helpers
    # ------------------------------------------------------------------

    def _resolve_model(self) -> str:
        model = self.config.get("model")
        if model:
            if model.startswith("http"):
                return self._download(model)
            path = expanduser(model)
            if not isfile(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            return path

        # Auto-resolve from ESPHome repository
        name = self.config.get("model_name", self.key_phrase.lower().replace(" ", "_"))
        version = int(self.config.get("model_version", 1))
        url = _model_url(name, version)
        return self._download(url)

    @staticmethod
    def _download(url: str) -> str:
        folder = join(xdg_data_home(), "microwakeword")
        # v1 and v2 models share a file name, so the URL keys the cache
        digest = hashlib.sha256(url.encode()).hexdigest()[:12]
        dest = join(folder, f"{digest}-{url.split('/')[-1]}")
        if not isfile(dest):
            LOG.info(f"Downloading microWakeWord model: {url}")
            os.makedirs(folder, exist_ok=True)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(r.content)
            LOG.info(f"Model saved to {dest}")
        return dest

    # ------------------------------------------------------------------
    # OVOS HotWordEngine contract
    # ------------------------------------------------------------------

    def update(self, chunk: bytes) -> None:
        """
        Process a raw 16-bit PCM audio chunk.

        Called by OVOS at ~50 ms intervals with 16 kHz mono int16 bytes.
        Sets an internal flag when the model fires.
        """
        if self._engine.is_detected(chunk):
            self._trigger_flag = True

    def found_wake_word(self) -> bool:
        """
        Return True if a wake-word event is pending and reset the flag.

        Called by OVOS after each :meth:`update` to check for detections.
        """
        if self._trigger_flag:
            self._trigger_flag = False
            self._engine.reset()
            return True
        return False
