# ovos-ww-plugin-microwakeword

This plugin adds wake-word detection to OpenVoiceOS. It wraps
[microWakeWord](https://github.com/kahrendt/microWakeWord) TFLite streaming
models from the [ESPHome ecosystem](https://github.com/esphome/micro-wake-word-models).

## Supported models

Models published at <https://github.com/esphome/micro-wake-word-models>:

| `model_name`  | Phrase         | v1  | v2  |
|---------------|----------------|-----|-----|
| `okay_nabu`   | Okay Nabu      | yes | yes |
| `hey_jarvis`  | Hey Jarvis     | yes | yes |
| `alexa`       | Alexa          | yes | yes |
| `hey_mycroft` | Hey Mycroft    | no  | yes |
| `vad`         | Voice activity | no  | yes |

The plugin also accepts any community `.tflite` model that follows the
microWakeWord input convention (1x1x40 int8 log-mel features).

## Installation

```bash
pip install ovos-ww-plugin-microwakeword
```

The package installs `ai-edge-litert` (on Linux x86_64) or `tflite-runtime`
(on other platforms) as a runtime dependency, along with `pymicro-features`
(the TFLite Micro audio frontend wrapper).

## Usage

Add the plugin to the `hotwords` section of `~/.config/mycroft/mycroft.conf`
(or `ovos.conf`), under your chosen wake word:

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
| `model`              | `str`   | *(auto)*     | Absolute path to a `.tflite` file, or an `https://` URL. This setting takes precedence over `model_name`. |
| `model_name`         | `str`   | `okay_nabu`  | Short name of an official ESPHome model. The plugin downloads it on first use. |
| `model_version`      | `int`   | `1`          | `1` or `2`. Selects the model subdirectory in the ESPHome repository. |
| `probability_cutoff` | `float` | `0.5`        | Dequantized probability threshold, in the range [0, 1]. A higher value gives fewer false positives. A lower value gives fewer missed detections. |
| `sliding_window_size`| `int`   | `10`         | Number of consecutive 10 ms frames whose average must exceed `probability_cutoff` before a detection fires. This mirrors the ESPHome `sliding_window_average_size` setting. |
| `refractory_frames`  | `int`   | `40`         | Frames to ignore after a detection (about 400 ms). This prevents double-fires. |

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
variables. Each sequential `interpreter.invoke()` call advances the internal
state automatically, so the plugin does not manage an external state tensor.
`interpreter.allocate_tensors()` resets the streaming state. The plugin
calls it from `reset()`.

### ESPHome model compatibility notes

- v1 models use the original microWakeWord architecture, with quantized int8
  input for the TFLite Micro audio frontend.
- v2 models use the same input convention. The plugin supports both
  transparently.
- A model must accept `[1, 1, 40] int8` input. Any model with a different
  input shape raises `ValueError` at load time.
- The audio frontend (`pymicro-features`) is the same C implementation that
  ESPHome uses for on-device inference.

## How to test

### Unit tests (no model required)

```bash
pytest tests/test_unit.py -v
```

All 16 unit tests use a mocked interpreter and pass without network access.

### End-to-end tests (downloads okay_nabu.tflite, requires edge-tts and ffmpeg)

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

The positive test skips rather than fails when TTS audio does not trigger
the model, because the model is trained on human voice. The negative test
("hello world") is a hard assertion.

## Related projects

Other OpenVoiceOS wake-word plugins:

- [ovos-ww-plugin-openWakeWord](https://github.com/OpenVoiceOS/ovos-ww-plugin-openWakeWord)
- [ovos-ww-plugin-precise-onnx](https://github.com/OpenVoiceOS/ovos-ww-plugin-precise-onnx)
- [ovos-ww-plugin-vosk](https://github.com/OpenVoiceOS/ovos-ww-plugin-vosk)
- [ovos-ww-plugin-wakeforge](https://github.com/OpenVoiceOS/ovos-ww-plugin-wakeforge)
- [ovos-ww-plugin-wakewordlab](https://github.com/OpenVoiceOS/ovos-ww-plugin-wakewordlab)

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

---

## License

Apache-2.0
