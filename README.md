# ovos-ww-plugin-microwakeword

OVOS wake-word plugin wrapping [microWakeWord](https://github.com/kahrendt/microWakeWord)
TFLite streaming models from the [ESPHome ecosystem](https://github.com/esphome/micro-wake-word-models).

## Supported models

Models published at <https://github.com/esphome/micro-wake-word-models>:

| `model_name`  | Phrase         | v1 | v2 |
|---------------|----------------|----|----|
| `okay_nabu`   | Okay Nabu      | ✓  | ✓  |
| `hey_jarvis`  | Hey Jarvis     | ✓  | ✓  |
| `alexa`       | Alexa          | ✓  | ✓  |
| `hey_mycroft` | Hey Mycroft    | –  | ✓  |
| `vad`         | Voice activity | –  | ✓  |

Any community-provided `.tflite` model that follows the microWakeWord input
convention (1×1×40 int8 log-mel features) is compatible.

## Installation

```bash
pip install ovos-ww-plugin-microwakeword
```

The package declares `ai-edge-litert` (Linux x86\_64) or `tflite-runtime`
(other platforms) as a runtime dependency alongside `pymicro-features`
(the TFLite Micro audio frontend wrapper).

## Configuration

In `~/.config/mycroft/mycroft.conf` (or `ovos.conf`), under the `hotwords`
section for your chosen keyword:

```json
{
  "hotwords": {
    "okay nabu": {
      "module": "ovos-ww-plugin-microwakeword",
      "model_name": "okay_nabu",
      "model_version": 1,
      "probability_cutoff": 0.5,
      "sliding_window_size": 10,
      "refractory_frames": 40
    }
  }
}
```

### Configuration reference

| Key                  | Type    | Default      | Description |
|----------------------|---------|--------------|-------------|
| `model`              | `str`   | *(auto)*     | Absolute path to a `.tflite` file, or an `https://` URL. Takes precedence over `model_name`. |
| `model_name`         | `str`   | `okay_nabu`  | Short name of an official ESPHome model. Auto-downloads on first use. |
| `model_version`      | `int`   | `1`          | `1` or `2` — selects the model subdirectory in the ESPHome repository. |
| `probability_cutoff` | `float` | `0.5`        | Dequantized probability threshold in [0, 1]. Higher → fewer false positives, lower → fewer missed detections. |
| `sliding_window_size`| `int`   | `10`         | Number of consecutive 10 ms frames whose average must exceed `probability_cutoff` before a detection fires. Mirrors ESPHome `sliding_window_average_size`. |
| `refractory_frames`  | `int`   | `40`         | Frames to ignore after a detection (≈ 400 ms) to prevent double-fires. |

## Technical details

### Audio pipeline

```
16 kHz int16 PCM  →  pymicro-features (TFLite Micro audio frontend)
                  →  40-dim log-mel feature slice per 10 ms frame
                  →  quantize to int8 (scale 0.102, zero-point −128)
                  →  TFLite interpreter (1×1×40 → 1×1 uint8)
                  →  dequantize → float probability
                  →  sliding window average ≥ cutoff → detection
```

### Model input signature

Inspected from `okay_nabu.tflite` (v1):

```
Input  tensor: serving_default_input_audio:0  shape=[1, 1, 40]  dtype=int8
               quantization: scale=0.10196, zero_point=-128
Output tensor: StatefulPartitionedCall:0      shape=[1, 1]       dtype=uint8
               quantization: scale=0.00390625, zero_point=0
```

The model embeds its streaming RNN/convolution state as TFLite resource
variables.  Each sequential `interpreter.invoke()` call advances the internal
state automatically — no external state tensor management is needed.
`interpreter.allocate_tensors()` resets the streaming state (called by
`reset()`).

### ESPHome model compatibility notes

- **v1 models** use the original microWakeWord architecture; quantized int8
  input with the TFLite Micro audio frontend.
- **v2 models** use the same input convention — the plugin supports both
  transparently.
- Models must accept `[1, 1, 40] int8` input; any model with a different
  input shape will raise `ValueError` at load time.
- The audio frontend (`pymicro-features`) is the exact same C implementation
  used by ESPHome's on-device inference.

## How to test

### Unit tests (no model required)

```bash
pytest tests/test_unit.py -v
```

All 16 unit tests use a mocked interpreter and pass without network access.

### End-to-end tests (downloads okay_nabu.tflite, requires edge-tts + ffmpeg)

```bash
pip install edge-tts
pytest tests/test_e2e.py -v -s
```

Expected output:

```
tests/test_e2e.py::TestE2EReal::test_negative_no_detection PASSED
[positive test] DETECTION FIRED on 'okay nabu' TTS audio.   ← or SKIPPED with max_prob info
tests/test_e2e.py::TestE2EReal::test_positive_detection PASSED
```

The positive test soft-fails (SKIP) rather than hard-fails when TTS audio
does not trigger the model, because the model is trained on human voice.
The negative test ("hello world") is a hard assertion.

## License

Apache-2.0

## Credits

Developed by [TigreGotico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org).

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) / [NLnet](https://nlnet.nl)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429),
through the European Commission's [Next Generation Internet](https://ngi.eu) programme.
