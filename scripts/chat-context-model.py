#!/usr/bin/env python3
"""Train and export a local recorded-reply cross-encoder from synthetic choices.

No remote inference, customer conversations, app/controller instantiation, or car
commands. Download the pinned base separately, then all operations are offline.
Models and large reports belong in the ignored build directory.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUIET = "No available recording fits this conversation. Stay quiet."
MAX_LENGTH = 192


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    if isinstance(path, dict):
        return path
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_inputs(catalogue_path, data_path):
    catalogue = load(catalogue_path)
    data = load(data_path)
    records = {row["id"]: row["text"] for row in catalogue["responses"]}
    if len(records) != len(catalogue["responses"]) or 0 in records:
        raise ValueError("Catalogue contains duplicate or reserved IDs")
    records[0] = QUIET
    aliases = collections.defaultdict(set)
    for ident, text in records.items():
        aliases[normalized(text)].add(ident)
    groups, contexts = set(), set()
    cases = []
    for group in data["groups"]:
        if group["id"] in groups or group["split"] not in ("train", "validation", "test"):
            raise ValueError("Duplicate group or invalid split")
        groups.add(group["id"])
        if not group["users"] or not group["preferredIDs"]:
            raise ValueError("Empty group")
        if not set(group["preferredIDs"] + group["rejectedIDs"]) <= records.keys():
            raise ValueError("Unknown recording label")
        if set(group["preferredIDs"]) & set(group["rejectedIDs"]):
            raise ValueError("Conflicting label")
        recent = []
        turns = []
        for role, value in group["history"][-8:]:
            if role == "mcqueen":
                if value not in records or value == 0:
                    raise ValueError("Invalid preceding recording")
                recent.append(value)
                turns.append("MCQUEEN: " + records[value])
            elif role == "user" and isinstance(value, str):
                turns.append("USER: " + value[:400])
            else:
                raise ValueError("Invalid preceding turn")
        excluded = set()
        for ident in recent[-5:]:
            excluded.update(aliases[normalized(records[ident])])
        choices = {}
        for index, ident in enumerate(group["preferredIDs"]):
            for alias in aliases[normalized(records[ident])]:
                if alias not in excluded:
                    choices[alias] = max(choices.get(alias, 0), 1.0 if index == 0 else 0.6)
        if not choices:
            raise ValueError("All teacher choices repeat preceding speech: " + group["id"])
        for index, user in enumerate(group["users"]):
            key = json.dumps([group["history"], user], sort_keys=True)
            if key in contexts:
                raise ValueError("Conversation appears more than once or leaks across splits")
            contexts.add(key)
            text = "Conversation with Lightning McQueen.\n" + "\n".join(turns)
            text += "\nLATEST USER: " + user[:500] + "\nChoose a suitable recorded reply."
            cases.append({"id": group["id"] + f"-{index + 1}", "groupID": group["id"],
                          "split": group["split"], "history": group["history"], "user": user,
                          "context": text, "preferred": choices, "rejected": group["rejectedIDs"],
                          "eligible": sorted(records.keys() - excluded),
                          "holdoutTopic": group.get("holdoutTopic", False)})
    return records, cases, data


def imported():
    # Keep data validation/tests independent of heavy optional training libraries.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return np, torch, AutoModelForSequenceClassification, AutoTokenizer


def tokenize(tokenizer, contexts, replies):
    # Discard oldest context first; retain the latest turn and entire recording.
    return tokenizer(contexts, replies, padding="max_length", truncation="only_first",
                     max_length=MAX_LENGTH, return_tensors="pt")


def predict(model, tokenizer, context, ids, records, device, batch_size=32):
    _, torch, _, _ = imported()
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            part = ids[start:start + batch_size]
            features = tokenize(tokenizer, [context] * len(part), [records[i] for i in part])
            logits = model(**{k: v.to(device) for k, v in features.items()}).logits.flatten()
            output.extend(logits.cpu().tolist())
    return output


def evaluate(model, tokenizer, cases, records, device, label):
    results = []
    for index, case in enumerate(cases):
        start = time.perf_counter()
        ids = case["eligible"]
        scores = predict(model, tokenizer, case["context"], ids, records, device)
        ranked = sorted(zip(ids, scores), key=lambda row: (-row[1], row[0]))
        chosen = ranked[0][0]
        results.append({"id": case["id"], "groupID": case["groupID"],
                        "history": case["history"], "user": case["user"],
                        "acceptableIDs": sorted(case["preferred"]), "selectedID": chosen,
                        "selectedText": records[chosen], "eligibleCount": len(ids),
                        "referenceHit": chosen in case["preferred"],
                        "top5ReferenceHit": any(i in case["preferred"] for i, _ in ranked[:5]),
                        "rejectedHit": chosen in case["rejected"],
                        "noMatchExpected": 0 in case["preferred"],
                        "holdoutTopic": case["holdoutTopic"],
                        "elapsedSeconds": time.perf_counter() - start,
                        "top": [{"id": i, "logit": s, "text": records[i]} for i, s in ranked[:5]]})
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(f"{label}: {index + 1}/{len(cases)} conversations scored", flush=True)
    positives = [r for r in results if not r["noMatchExpected"]]
    quiet = [r for r in results if r["noMatchExpected"]]
    metric = lambda rows: sum(r["referenceHit"] for r in rows) / max(1, len(rows))
    metrics = {"count": len(results), "referenceHits": sum(r["referenceHit"] for r in results),
               "positiveCount": len(positives), "positiveReferenceRate": metric(positives),
               "noMatchCount": len(quiet), "noMatchRate": metric(quiet),
               "balancedReferenceRate": (metric(positives) + metric(quiet)) / 2,
               "top5ReferenceHits": sum(r["top5ReferenceHit"] for r in results),
               "selectedQuietCount": sum(r["selectedID"] == 0 for r in results),
               "rejectedHits": sum(r["rejectedHit"] for r in results),
               "medianSeconds": sorted(r["elapsedSeconds"] for r in results)[len(results) // 2]}
    print(label, json.dumps(metrics), flush=True)
    return {"metrics": metrics, "results": results}


def train(args):
    np, torch, AutoModel, AutoTokenizer = imported()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    records, cases, data = read_inputs(args.catalogue, args.data)
    training = [c for c in cases if c["split"] == "train"]
    validation = [c for c in cases if c["split"] == "validation"]
    test = [c for c in cases if c["split"] == "test"]
    if not training or not validation or not test:
        raise ValueError("All three splits are required")
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    tokenizer.truncation_side = "left"
    model = AutoModel.from_pretrained(args.initial or args.base, local_files_only=True, use_safetensors=True,
                                     trust_remote_code=False, attn_implementation="eager")
    probe = tokenize(tokenizer, [training[0]["context"]] * 4, [records[i] for i in (0, 202, 694, 745)])
    model.eval()
    with torch.inference_mode():
        cpu = model(**probe).logits
    # Moving parameters inside inference_mode marks them as inference-only
    # tensors, which cannot participate in the subsequent training backward pass.
    model.to(device)
    with torch.inference_mode():
        native = model(**{k: v.to(device) for k, v in probe.items()}).logits.cpu()
    error = float((cpu - native).abs().max())
    if error > 0.002:
        raise RuntimeError(f"Device numerical check failed: {error}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schemaVersion": 1, "labelSource": data["labelSource"], "device": device,
              "catalogueSHA256": sha(args.catalogue), "datasetSHA256": sha(args.data),
              "trainerSHA256": sha(__file__), "seed": args.seed, "maxLength": MAX_LENGTH,
              "trainingConversations": len(training), "validationConversations": len(validation),
              "testConversations": len(test), "parameters": sum(p.numel() for p in model.parameters()),
              "positiveRecordingCoverage": len({i for c in training for i in c["preferred"] if i}),
              "cpuDeviceProbeMaxError": error, "config": vars(args).copy(), "epochs": [],
              "caveat": "Assistant-authored synthetic preference agreement, not independently rated conversational accuracy. Fixed catalogue, mostly familiar topics. No phone runtime claim."}
    report["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["config"].items()}
    report["initialValidation"] = evaluate(model, tokenizer, validation, records, device, "Initial validation")
    mined = {}
    if args.hard_negatives:
        for index, case in enumerate(training):
            ids = case["eligible"]
            scores = predict(model, tokenizer, case["context"], ids, records, device)
            ranked = sorted(zip(ids, scores), key=lambda row: (-row[1], row[0]))
            mined[case["id"]] = [i for i, _ in ranked if i not in case["preferred"]][:args.hard_negatives]
            if (index + 1) % 25 == 0 or index + 1 == len(training):
                print(f"Mining training-only competitors: {index+1}/{len(training)}", flush=True)
        save(args.output / "mined-training-competitors.json", {
            "datasetSHA256": report["datasetSHA256"], "initialWeightsSHA256": sha((args.initial or args.base) / "model.safetensors"),
            "trainingOnly": True, "choices": mined,
            "meaning": "Lower teacher preference than supplied suitable replies, not independent judgments of factual correctness."})
        report["minedCompetitorSHA256"] = sha(args.output / "mined-training-competitors.json")
    save(args.output / "report.json", report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = math.ceil(len(training) / args.conversations_per_batch) * args.epochs
    step = 0; best = -1.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        ordering = list(training); random.Random(args.seed + epoch).shuffle(ordering)
        rng = random.Random(args.seed + epoch + 1000)
        losses = []; sampled = set()
        model.train()
        for offset in range(0, len(ordering), args.conversations_per_batch):
            batch = ordering[offset:offset + args.conversations_per_batch]
            contexts, replies, spans = [], [], []
            for case in batch:
                positive = sorted(case["preferred"])
                pool = [i for i in case["eligible"] if i not in case["preferred"]]
                hard = [i for i in case["rejected"] + mined.get(case["id"], []) if i in pool]
                if 0 in pool: hard.append(0)
                negative = list(dict.fromkeys(hard))
                additional = [i for i in pool if i not in negative]
                negative += rng.sample(additional, min(args.negatives, len(additional)))
                ids = positive + negative
                sampled.update(ids)
                start = len(contexts)
                contexts.extend([case["context"]] * len(ids)); replies.extend(records[i] for i in ids)
                spans.append((start, len(ids), [case["preferred"].get(i, 0.0) for i in ids],
                              [i in case["rejected"] for i in ids]))
            features = tokenize(tokenizer, contexts, replies)
            features = {k: v.to(device) for k, v in features.items()}
            logits = model(**features).logits.flatten()
            group_losses = []
            for start, length, weights, rejected in spans:
                scores = logits[start:start + length]
                target = torch.tensor(weights, dtype=torch.float32, device=device)
                distribution = target / target.sum()
                preference = -(distribution * torch.log_softmax(scores, dim=0)).sum()
                # Balanced pointwise preference regularization prevents the many
                # sampled lower-preference replies overwhelming positive choices.
                good = target > 0
                pointwise = (torch.nn.functional.softplus(-scores[good]).mean() +
                             torch.nn.functional.softplus(scores[~good]).mean()) / 2
                # Explicit contradictions get a separate margin. Unlisted
                # candidates remain lower-preference, not categorical errors.
                bad = torch.tensor(rejected, dtype=torch.bool, device=device)
                rejection = scores.sum() * 0
                if args.rejection_weight and bad.any():
                    rejection = torch.nn.functional.softplus(
                        1.0 - (scores[good][:, None] - scores[bad][None, :])).mean()
                group_losses.append(preference + 0.2 * pointwise + args.rejection_weight * rejection)
            loss = torch.stack(group_losses).mean()
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step += 1
            warmup = max(1, int(total_steps * 0.1))
            scale = step / warmup if step <= warmup else max(0.05, (total_steps - step) / (total_steps - warmup))
            for param in optimizer.param_groups: param["lr"] = args.learning_rate * scale
            optimizer.step(); losses.append(float(loss.detach().cpu()))
            if len(losses) % 25 == 0:
                print(f"Epoch {epoch+1}: {offset+len(batch)}/{len(ordering)} conversations; loss {sum(losses[-25:])/25:.4f}", flush=True)
        val = evaluate(model, tokenizer, validation, records, device, f"Epoch {epoch+1} validation")
        metric = val["metrics"]["balancedReferenceRate"]
        report["epochs"].append({"epoch": epoch+1, "meanLoss": sum(losses)/len(losses),
                                 "sampledCandidateCoverage": len(sampled), "validation": val})
        if metric > best:
            best = metric; report["chosenEpoch"] = epoch+1
            model.save_pretrained(args.output / "trained", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "trained")
        save(args.output / "report.json", report)
        model.train()
    report["trainingSeconds"] = time.perf_counter() - started
    # Iterative work can defer all final-test inference until configuration is
    # frozen. Validation alone chooses checkpoints and curriculum changes.
    report["testDeferred"] = args.skip_test
    if not args.skip_test:
        trained = AutoModel.from_pretrained(args.output / "trained", local_files_only=True,
                                           use_safetensors=True, attn_implementation="eager").to(device)
        report["trainedTest"] = evaluate(trained, tokenizer, test, records, device, "Trained test")
        baseline = AutoModel.from_pretrained(args.base, local_files_only=True, use_safetensors=True,
                                            attn_implementation="eager").to(device)
        report["baseTest"] = evaluate(baseline, tokenizer, test, records, device, "Base test")
    report["trainedWeightsSHA256"] = sha(args.output / "trained/model.safetensors")
    report["completed"] = True
    save(args.output / "report.json", report)
    print("Completed local training; final tests deferred." if args.skip_test else
          "Completed local training and fixed-test comparison.", flush=True)


def evaluate_checkpoint(args):
    _, torch, AutoModel, AutoTokenizer = imported()
    torch.set_num_threads(6)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    records, cases, _ = read_inputs(args.catalogue, args.data)
    selected = [c for c in cases if c["split"] == args.split]
    checkpoint = args.checkpoint or args.output / "trained"
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True,
                                     trust_remote_code=False, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    tokenizer.truncation_side = "left"
    report = evaluate(model, tokenizer, selected, records, device, "Fixed " + args.split)
    report.update({"checkpoint": str(checkpoint), "weightsSHA256": sha(checkpoint / "model.safetensors"),
                   "catalogueSHA256": sha(args.catalogue), "datasetSHA256": sha(args.data),
                   "evaluatorSHA256": sha(__file__), "device": device, "split": args.split,
                   "caveat": "Synthetic teacher preferences, not independently rated conversational accuracy."})
    save(args.evaluation_report or args.output / ("evaluation-" + args.split + ".json"), report)


def export(args):
    np, torch, AutoModel, AutoTokenizer = imported()
    import coremltools as ct
    import onnxruntime as ort
    report = load(args.output / "report.json")
    if (not report.get("completed") or report["datasetSHA256"] != sha(args.data)
        or report["catalogueSHA256"] != sha(args.catalogue)
        or report["trainedWeightsSHA256"] != sha(args.output / "trained/model.safetensors")):
        raise ValueError("Completed matching training report required")
    records, cases, _ = read_inputs(args.catalogue, args.data)
    model = AutoModel.from_pretrained(args.output / "trained", local_files_only=True,
                                     use_safetensors=True, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.output / "trained", local_files_only=True)
    tokenizer.truncation_side = "left"

    class Scorer(torch.nn.Module):
        def __init__(self, model): super().__init__(); self.model = model
        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, return_dict=False)[0]

    scorer = Scorer(model).eval()
    test_cases = [c for c in cases if c["split"] == "test"]
    selected_probes = [test_cases[round(i * (len(test_cases)-1) / 7)] for i in range(8)]
    probes = [(c["context"], i) for c in selected_probes
              for i in [0, *list(c["preferred"])[:1], 202, 694]]
    features = tokenize(tokenizer, [c for c, _ in probes], [records[i] for _, i in probes])
    # Transformers uses float32's -3.4e38 for masked attention. Converting that
    # constant to float16 overflows and changes Core ML predictions. -10000
    # suppresses masked softmax weights identically and is representable in fp16.
    # Verify this export-only substitution against untouched trained weights.
    with torch.inference_mode():
        expected = scorer(*(features[n] for n in ("input_ids", "attention_mask", "token_type_ids"))).numpy()
    def finite_encoder_mask(self, attention_mask, input_shape, device=None, dtype=None):
        if attention_mask.dim() != 2 or self.config.is_decoder:
            raise ValueError("This export requires a two-dimensional BERT encoder mask")
        mask = attention_mask[:, None, None, :].to(dtype=dtype or self.dtype)
        return (1.0 - mask) * -10000.0
    model.bert.get_extended_attention_mask = types.MethodType(finite_encoder_mask, model.bert)
    with torch.inference_mode():
        bounded = scorer(*(features[n] for n in ("input_ids", "attention_mask", "token_type_ids"))).numpy()
    mask_error = float(np.max(np.abs(expected - bounded)))
    if mask_error > 0.0001:
        raise RuntimeError(f"Finite attention-mask substitution changes predictions: {mask_error}")
    example = tuple(features[k][:2] for k in ("input_ids", "attention_mask", "token_type_ids"))
    names = ["input_ids", "attention_mask", "token_type_ids"]
    exports = args.output / "exports"; exports.mkdir(exist_ok=True)
    onnx_path = exports / "McQueenReplyRanker.onnx"
    torch.onnx.export(scorer, example, onnx_path, input_names=names, output_names=["reply_score"],
                      dynamic_axes={**{n: {0: "batch"} for n in names}, "reply_score": {0: "batch"}},
                      opset_version=17, dynamo=False)
    traced = torch.jit.trace(scorer, example)
    batch = ct.RangeDim(lower_bound=1, upper_bound=32, default=16)
    converted = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                           inputs=[ct.TensorType(name=n, shape=(batch, MAX_LENGTH), dtype=np.int32) for n in names],
                           outputs=[ct.TensorType(name="reply_score")], compute_precision=ct.precision.FLOAT16)
    converted.author = "Red Thunder"
    converted.short_description = "Recorded McQueen reply relevance, trained from synthetic contextual preferences."
    converted.version = "context-" + str(load(args.data)["schemaVersion"])
    converted.user_defined_metadata.update({"catalogueSHA256": report["catalogueSHA256"],
                                           "datasetSHA256": report["datasetSHA256"],
                                           "maxLength": str(MAX_LENGTH), "labelSource": "Codex synthetic choices"})
    ml_path = exports / "McQueenReplyRanker.mlpackage"; converted.save(ml_path)
    tokenizer.save_pretrained(exports / "tokenizer")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {n: features[n].numpy() for n in names})[0]
    onnx_error = float(np.max(np.abs(expected - actual)))
    reloaded = ct.models.MLModel(str(ml_path))
    output = reloaded.predict({n: features[n].numpy().astype(np.int32) for n in names})["reply_score"]
    coreml_error = float(np.max(np.abs(expected - output)))
    if onnx_error > 0.002 or coreml_error > 0.15:
        raise RuntimeError(f"Export parity failed: ONNX {onnx_error}, Core ML {coreml_error}")
    save(exports / "manifest.json", {"schemaVersion": 1, "maxLength": MAX_LENGTH,
         "catalogueSHA256": report["catalogueSHA256"], "datasetSHA256": report["datasetSHA256"],
         "weightsSHA256": report["trainedWeightsSHA256"], "noMatchText": QUIET,
         "contextFormat": "\n".join(["Conversation with Lightning McQueen.", "ROLE: text...",
                                     "LATEST USER: text", "Choose a suitable recorded reply."]),
         "padding": "right", "truncate": "oldest context first; preserve reply; only_first",
         "inputs": names, "output": "reply_score", "batchRange": [1, 32],
         "onnxSHA256": sha(onnx_path), "onnxMaxAbsoluteError": onnx_error,
         "coremlMaxAbsoluteError": coreml_error, "parityProbeCount": len(probes),
         "exporterSHA256": sha(__file__), "attentionMaskValue": -10000,
         "attentionMaskSubstitutionMaxError": mask_error,
         "modelLicense": "Apache-2.0", "sourceModel": load(args.base.parent / "base-provenance.json")})
    print(f"Exported and reloaded both runtimes; errors ONNX={onnx_error:.6f}, Core ML={coreml_error:.6f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["validate", "train", "export", "evaluate"])
    p.add_argument("--catalogue", type=Path, default=ROOT / "SpheroMcQueen/ChatResponses.json")
    p.add_argument("--data", type=Path, default=ROOT / "docs/training/chat-context-teacher-v1.json")
    p.add_argument("--base", type=Path, default=ROOT / "build/chat-context-model/base-model")
    p.add_argument("--output", type=Path, default=ROOT / "build/chat-context-model/run-v1")
    p.add_argument("--initial", type=Path, help="Optional earlier trained checkpoint for a second curriculum stage")
    p.add_argument("--hard-negatives", type=int, default=0, help="Mine this many competitors per training conversation")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--conversations-per-batch", type=int, default=2)
    p.add_argument("--negatives", type=int, default=20)
    p.add_argument("--rejection-weight", type=float, default=0.0,
                   help="Additional margin for explicitly rejected replies; zero preserves v1 loss")
    p.add_argument("--skip-test", action="store_true", help="Defer final-test evaluation during iterations")
    p.add_argument("--checkpoint", type=Path, help="Weights for standalone fixed evaluation")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--evaluation-report", type=Path, help="Standalone evaluation output path")
    args = p.parse_args()
    if args.command == "validate":
        records, cases, _ = read_inputs(args.catalogue, args.data)
        print(json.dumps({"recordings": len(records)-1,
              "splits": dict(collections.Counter(c["split"] for c in cases)),
              "datasetSHA256": sha(args.data)}, indent=2))
    elif args.command == "train": train(args)
    elif args.command == "evaluate": evaluate_checkpoint(args)
    else: export(args)


if __name__ == "__main__":
    main()
