# Reproduce or modify the model

Run from this repository root. Install requirements from README. The pinned base is Apache-2.0; download it without remote model code:

```python
from huggingface_hub import snapshot_download
snapshot_download("cross-encoder/ms-marco-MiniLM-L6-v2", revision="233902d25c440f23af6f7d6e94d2946bac0bee0a", local_dir="build/chat-context-model/base-model", allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"])
```

The frozen model used four curriculum stages. These commands reproduce the recipe; GPU kernels and library/platform differences can change exact weights. Use released hashes to verify the shipped artifacts. All training after the base download runs locally.

```sh

HF_HUB_OFFLINE=1 python3 scripts/chat-context-model.py train --data docs/training/chat-context-teacher-v1.json --output build/chat-context-model/run-v1 --epochs 4 --learning-rate 2e-05 --negatives 20 --seed 1729

HF_HUB_OFFLINE=1 python3 scripts/chat-context-model.py train --data docs/training/chat-context-teacher-v1.json --output build/chat-context-model/run-v2 --initial build/chat-context-model/run-v1/trained --epochs 6 --learning-rate 1.5e-05 --hard-negatives 16 --negatives 12 --seed 1729

HF_HUB_OFFLINE=1 python3 scripts/chat-context-model.py train --data docs/training/chat-context-teacher-v2.json --output build/chat-context-model/run-v3 --initial build/chat-context-model/run-v2/trained --epochs 6 --learning-rate 1e-05 --hard-negatives 12 --negatives 16 --rejection-weight 0.5 --seed 1729 --skip-test

HF_HUB_OFFLINE=1 python3 scripts/chat-context-model.py train --data docs/training/chat-context-teacher-v3.json --output build/chat-context-model/run-v4 --initial build/chat-context-model/run-v3/trained --epochs 3 --learning-rate 7.5e-06 --hard-negatives 12 --negatives 16 --rejection-weight 0.5 --seed 1729 --skip-test

python3 scripts/chat-context-model.py export --output build/chat-context-model/run-v4
python3 scripts/export-chat-runtime-flex.py
```

The shape-only exporter verifies every ONNX initializer against the frozen export. The app uses the original fixed-192 Core ML package from `exports/`, together with flexible ONNX from `runtime-flex/`. Do not substitute the flexible Core ML package: its graph and weight layout differ from the app’s bundled graph. Use the included parity checker before publishing. The release has known fixed hashes; a fresh Core ML conversion can change serialized bytes.

For an app override, copy `manifests/custom.example.json`, replace each platform URL/length/SHA-256, and keep compatibility fields unchanged. iOS downloads only the Core ML package’s `weight.bin`; Android downloads the ONNX file. The app rejects redirects, hash/length mismatches and incompatible metadata. See the website guide.

Generate the app manifest with `python3 scripts/make-app-manifest.py --ios-package PATH/McQueenReplyRanker.mlpackage --android PATH/model.onnx --ios-url https://your-host.example/weights.bin --android-url https://your-host.example/model.onnx --revision 2 --output manifest.json`. The helper rejects a Core ML graph/package that differs from the app compatibility baseline.
