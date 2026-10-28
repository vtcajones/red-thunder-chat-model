# Model card

## What it does

The 22,713,601-parameter model fine-tunes `cross-encoder/ms-marco-MiniLM-L6-v2` for English contextual reply selection. Input is a conversation/reply pair; output is one `reply_score`. Score the full eligible catalogue plus the explicit no-match candidate, then choose the highest finite score. Repeated recording IDs and identical recent spoken text are excluded by the host.

- Base revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`, Apache-2.0.
- Frozen checkpoint SHA-256: `6224e69ea81c6099eee02a4b34397216376faa8624a33f7f7583f103e5e1f426`.
- Dataset v3 SHA-256: `ed035f08f09878b421e23907741e3f36d6706efd7642053403dd1e39755cae44`.
- Catalogue SHA-256: `58b4a73b2d22e7bf901fca326efc0ccd34d2c835c794e3f597eab4fab2a3c62e`.
- Inputs: `input_ids`, `attention_mask`, `token_type_ids`; output: `reply_score`.
- WordPiece vocabulary is pinned. Context/reply length is at most 192 tokens.
- Core ML uses int32 inputs, batches 1–32, a fixed 192-token sequence and the bundled graph/weight layout. ONNX uses int64 inputs and flexible batch/sequence axes. Android removes only masked right padding, after the same truncation.

## Data and evaluation

Codex authored synthetic conversation preferences after reading the 318-line catalogue. The final stage uses 1,459 training conversations and 70 validation conversations; 80 fixed test conversations are retained separately. Complete conversations are split together. Earlier stages and regression cases are retained. No private customer conversations or microphone audio were used.

Labels express incomplete teacher preferences, not independent human judgments. Some catalogue entries have no positive training examples. Familiar topics and characters appear across splits. Validation chose the frozen checkpoint before fresh-test evaluation. This does not establish general conversational accuracy, safety, physical-car behavior or phone performance.

## Limits

This is a selector for a fixed catalogue, not an LLM or a translation/voice-cloning model. It can choose an irrelevant or contradictory line. It cannot invent a missing answer. Training changes may improve selection; changing the vocabulary, catalogue or Core ML graph requires corresponding app changes rather than a weights-only override. The model must never be used to authorize driving or other car operations.
