#!/usr/bin/env python3
"""Verify complete-catalogue export choices against a saved PyTorch evaluation.

All inference is local. This is conversion verification, not conversation quality
or a hardware benchmark on iPhone/Android. The input is a synthetic research set.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('context_helpers', ROOT / 'scripts/chat-context-model.py')
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='Export directory')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--catalogue', type=Path, default=ROOT / 'SpheroMcQueen/ChatResponses.json')
    p.add_argument('--backend', choices=['coreml', 'onnx', 'both'], default='both')
    args = p.parse_args()
    manifest = HELPERS.load(args.model / 'manifest.json')
    expected = HELPERS.load(args.evaluation)
    for key, actual in [('datasetSHA256', HELPERS.sha(args.data)),
                        ('catalogueSHA256', HELPERS.sha(args.catalogue)),
                        ('weightsSHA256', manifest['weightsSHA256'])]:
        if expected[key] != actual or manifest[key] != actual:
            raise ValueError('Mismatched evaluation/export inputs: ' + key)
    records, cases, _ = HELPERS.read_inputs(args.catalogue, args.data)
    cases = {c['id']: c for c in cases if c['split'] == expected['split']}
    if cases.keys() != {r['id'] for r in expected['results']}:
        raise ValueError('Missing/extra evaluation conversations')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.truncation_side = 'left'
    backends = ['coreml', 'onnx'] if args.backend == 'both' else [args.backend]
    report = {'scope': 'Mac export selection parity, not independently rated quality or phone timing',
              'manifest': manifest, 'evaluationSHA256': HELPERS.sha(args.evaluation),
              'verifierSHA256': HELPERS.sha(__file__), 'backends': {}}
    for backend in backends:
        if backend == 'coreml':
            import coremltools as ct
            runtime = ct.models.MLModel(str(args.model / 'McQueenReplyRanker.mlpackage'))
        else:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 6
            opts.inter_op_num_threads = 1
            runtime = ort.InferenceSession(str(args.model / 'McQueenReplyRanker.onnx'), opts,
                                           providers=['CPUExecutionProvider'])
        results = []
        for reference in expected['results']:
            case = cases[reference['id']]
            scores = []
            started = time.perf_counter()
            for start in range(0, len(case['eligible']), 16):
                ids = case['eligible'][start:start + 16]
                tokens = HELPERS.tokenize(tokenizer, [case['context']] * len(ids), [records[i] for i in ids])
                features = {name: tokens[name].numpy() for name in manifest['inputs']}
                if backend == 'coreml':
                    values = runtime.predict({k: v.astype(np.int32) for k,v in features.items()})[manifest['output']]
                else:
                    values = runtime.run([manifest['output']], features)[0]
                if not np.isfinite(values).all():
                    raise RuntimeError('Nonfinite exported logits')
                scores.extend(zip(ids, values.flatten().tolist()))
            choice = sorted(scores, key=lambda row: (-row[1], row[0]))[0][0]
            results.append({'id': case['id'], 'torchID': reference['selectedID'], 'exportID': choice,
                            'sameID': choice == reference['selectedID'],
                            'sameWords': records[choice] == records[reference['selectedID']],
                            'referenceHit': choice in case['preferred'],
                            'elapsedSeconds': time.perf_counter() - started})
            if len(results) % 10 == 0:
                print(f'{backend}: {len(results)}/{len(cases)} complete-catalogue choices checked', flush=True)
        metrics = {'count': len(results), 'sameID': sum(r['sameID'] for r in results),
                   'sameWords': sum(r['sameWords'] for r in results),
                   'referenceHits': sum(r['referenceHit'] for r in results)}
        report['backends'][backend] = {'metrics': metrics, 'results': results}
        HELPERS.save(args.report, report)
        print(backend, json.dumps(metrics), flush=True)
    if any(b['metrics']['sameWords'] != b['metrics']['count'] for b in report['backends'].values()):
        raise RuntimeError('An exported backend changed a selected reply; inspect the report')


if __name__ == '__main__':
    main()
