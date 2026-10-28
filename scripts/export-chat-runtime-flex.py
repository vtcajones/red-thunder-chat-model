#!/usr/bin/env python3
"""Re-export the frozen checkpoint with flexible sequence length, without training.

Only masked right padding may be removed at runtime, after the unchanged
192-token context truncation. Verification is separate; this does not install
or replace the app's model assets.
"""
import argparse
import hashlib
import json
import os
import shutil
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = '6224e69ea81c6099eee02a4b34397216376faa8624a33f7f7583f103e5e1f426'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'build/chat-context-model/run-v4')
    p.add_argument('--output', type=Path, default=ROOT/'build/chat-context-model/run-v4/runtime-flex')
    args = p.parse_args()
    if args.output.resolve() == (args.source/'exports').resolve():
        raise ValueError('Do not overwrite the frozen export')
    assert sha(args.source/'trained/model.safetensors') == WEIGHTS
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    import torch
    import onnx
    import coremltools as ct
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(2)
    model = AutoModelForSequenceClassification.from_pretrained(args.source/'trained', local_files_only=True,
        trust_remote_code=False, use_safetensors=True, attn_implementation='eager').eval()
    def finite_mask(self, attention_mask, input_shape, device=None, dtype=None):
        assert attention_mask.dim() == 2 and not self.config.is_decoder
        return (1.0 - attention_mask[:, None, None, :].to(dtype=dtype or self.dtype)) * -10000.0
    model.bert.get_extended_attention_mask = types.MethodType(finite_mask, model.bert)
    class Scorer(torch.nn.Module):
        def __init__(self, base): super().__init__(); self.model = base
        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, return_dict=False)[0]
    scorer = Scorer(model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.source/'exports/tokenizer', local_files_only=True, trust_remote_code=False)
    tokenizer.truncation_side = 'left'
    features = tokenizer(['Conversation with Lightning McQueen.\n\nLATEST USER: Hello!\nChoose a suitable recorded reply.']*2,
        ['Hey buddy!', 'No available recording fits this conversation. Stay quiet.'],
        padding='max_length', truncation='only_first', max_length=192, return_tensors='pt')
    names = ['input_ids', 'attention_mask', 'token_type_ids']
    example = tuple(features[n] for n in names)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'McQueenReplyRanker.onnx'
    torch.onnx.export(scorer, example, target, input_names=names, output_names=['reply_score'],
        dynamic_axes={**{n:{0:'batch',1:'sequence'} for n in names}, 'reply_score':{0:'batch'}},
        opset_version=17, dynamo=False)
    # Every trained initializer must be byte-identical to the frozen ONNX export.
    def initializers(path):
        graph = onnx.load(path)
        return {t.name: (list(t.dims), t.data_type, hashlib.sha256(t.raw_data).hexdigest()) for t in graph.graph.initializer}
    before = initializers(args.source/'exports/McQueenReplyRanker.onnx')
    after = initializers(target)
    assert before == after, 'Frozen weights changed during shape-only export'
    traced = torch.jit.trace(scorer, example)
    batch = ct.RangeDim(lower_bound=1, upper_bound=32, default=16)
    sequence = ct.RangeDim(lower_bound=1, upper_bound=192, default=192)
    converted = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS16,
        inputs=[ct.TensorType(name=n, shape=(batch, sequence), dtype=np.int32) for n in names],
        outputs=[ct.TensorType(name='reply_score')], compute_precision=ct.precision.FLOAT16)
    converted.author = 'Red Thunder'
    converted.short_description = 'Frozen contextual recording ranker; flexible masked-padding length.'
    converted.version = 'run-v4-runtime-flex'
    converted.user_defined_metadata.update({'weightsSHA256':WEIGHTS,'maxLength':'192','optimization':'masked right padding only'})
    converted.save(str(args.output/'McQueenReplyRanker.mlpackage'))
    for name in ['tokenizer','LICENSE-base-model.txt','base-model-card.md','base-provenance.json']:
        src = args.source/'exports'/name; dst = args.output/name
        if src.is_dir(): shutil.copytree(src,dst,dirs_exist_ok=True)
        else: shutil.copy2(src,dst)
    manifest = json.loads((args.source/'exports/manifest.json').read_text())
    manifest.update({'schemaVersion':2,'sequenceRange':[1,192], 'noNewTraining':True,
        'optimization':'remove masked right padding only, after frozen 192-token truncation',
        'sourceONNXSHA256':sha(args.source/'exports/McQueenReplyRanker.onnx'),
        'onnxSHA256':sha(target),'initializerParityCount':len(before),'initializerByteParity':True,
        'runtimeExporterSHA256':sha(__file__), 'verification':'pending separate fixed-case logit and selection checks'})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(args.output),'initializerByteParity':True,'onnxSHA256':manifest['onnxSHA256']}),flush=True)

if __name__ == '__main__': main()
