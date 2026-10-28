#!/usr/bin/env python3
"""Try an exported selector locally. Reads one conversation, sends no car commands.

Input JSON: {"history": [["user", "words"], ["mcqueen", 754]], "user": "latest words"}
No transcript file, log, or remote request is created by this command.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("context_model", ROOT / "scripts/chat-context-model.py")
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "build/chat-context-model/run-v4/exports")
    parser.add_argument("--backend", choices=["coreml", "onnx"], default="coreml")
    parser.add_argument("--catalogue", type=Path, default=ROOT / "SpheroMcQueen/ChatResponses.json")
    args = parser.parse_args()
    manifest = HELPERS.load(args.model / "manifest.json")
    if HELPERS.sha(args.catalogue) != manifest["catalogueSHA256"]:
        raise ValueError("This model was trained for a different recording catalogue")
    conversation = HELPERS.load(args.conversation)
    if not isinstance(conversation.get("user"), str) or not conversation["user"].strip():
        raise ValueError("Provide nonempty latest-user text")
    # Labels here are unused placeholders. Reuse the exact training formatter
    # and recent-spoken-text exclusion without writing a temporary transcript.
    data = {"groups": [{"id": "query", "split": "test", "history": conversation.get("history", []),
                         "users": [conversation["user"]], "preferredIDs": [0], "rejectedIDs": []}]}
    records, cases, _ = HELPERS.read_inputs(args.catalogue, data)
    case = cases[0]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.truncation_side = "left"
    if args.backend == "coreml":
        import coremltools as ct
        runtime = ct.models.MLModel(str(args.model / "McQueenReplyRanker.mlpackage"))
    else:
        import onnxruntime as ort
        runtime = ort.InferenceSession(str(args.model / "McQueenReplyRanker.onnx"),
                                       providers=["CPUExecutionProvider"])
    scores = []
    for start in range(0, len(case["eligible"]), 16):
        ids = case["eligible"][start:start + 16]
        features = HELPERS.tokenize(tokenizer, [case["context"]] * len(ids), [records[i] for i in ids])
        inputs = {name: features[name].numpy() for name in manifest["inputs"]}
        if args.backend == "coreml":
            output = runtime.predict({k: v.astype(np.int32) for k, v in inputs.items()})["reply_score"]
        else:
            output = runtime.run(["reply_score"], inputs)[0]
        scores.extend(zip(ids, output.flatten().tolist()))
    ranked = sorted(scores, key=lambda row: (-row[1], row[0]))
    choice = ranked[0][0]
    print(json.dumps({"selectedID": choice, "spokenText": records[choice] if choice else None,
                     "eligibleRecordings": len(case["eligible"]) - 1,
                     "top": [{"id": i, "score": score, "text": records[i]} for i, score in ranked[:5]]}, indent=2))


if __name__ == "__main__":
    main()
