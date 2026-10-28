# Red Thunder Chat model

A small offline cross-encoder that chooses a spoken reply from an existing toy’s recording catalogue. It scores each eligible line against the conversation and can choose silence. It does not generate speech or control the car.

This repository contains the training and export code, synthetic training preferences, recording-ID/transcript reference and model provenance. The first release includes the exact trained checkpoint and Core ML/ONNX runtime exports used by Red Thunder’s current development app. The app’s new system-settings override is still in development; this repository is usable independently.

## Downloads

Get the [v1.0.0 release](https://github.com/vtcajones/red-thunder-chat-model/releases/tag/v1.0.0):

- `checkpoint.zip`: Transformers checkpoint, tokenizer and source provenance.
- `runtime-exports.zip`: Core ML package, self-contained ONNX model, tokenizer, runtime manifest and license.
- `SHA256SUMS`: checksums for the release archives.

The app-compatible manifest is [`manifests/red-thunder-v1.json`](manifests/red-thunder-v1.json). Its immutable model URLs point to the existing public download service. GitHub release links redirect and cannot be pasted into the app as direct model-file URLs. Download the artifacts and host them on a direct HTTPS endpoint when publishing your own compatible weights.

[App override and manifest guide](https://redthunderremote.com/models/chat/guide/) · [Model card](MODEL_CARD.md) · [Training](TRAINING.md) · [License](LICENSE)

## Try it locally

Python 3.12 was used for the frozen run. Core ML export/inference requires macOS; ONNX inference is cross-platform.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 scripts/chat-context-model.py validate
python3 -m unittest discover -s scripts/tests -p test_chat_context_model.py
```

Unzip `runtime-exports.zip` to `build/chat-context-model/run-v4/exports/`, then:

```sh
python3 scripts/query-chat-context-model.py \
  --backend onnx --conversation docs/training/chat-context-example.json
```

The query prints a selected recording ID and text. It sends no audio or car commands and makes no remote inference call. Models/tokenizers are loaded with remote code disabled. Download the pinned base separately before training.

## License and scope

Original code, synthetic training contributions and fine-tuned weights use Apache-2.0, retaining upstream attribution. The recording catalogue is an interoperability reference: source dialogue and trademarks retain their respective ownership. See NOTICE. No car audio, firmware, artwork, credentials or customer logs are included.

The classifier’s output is only a proposed recording ID. Host applications must retain their own playback allowlist, user consent, permissions and connection/lifecycle checks. This repository does not change Red Thunder’s Pro policy.
