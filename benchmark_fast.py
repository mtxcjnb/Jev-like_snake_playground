"""Paired measurement and exact prepared-input checks for the faster request path."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from PIL import Image

from benchmark import stats
from qwen_choice import QwenChoice
from qwen_choice_fast import FastQwenChoice


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.5-4B')
    ap.add_argument('--local-files-only', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('results/local-fast'))
    ap.add_argument('--compile-language', action='store_true')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=FastQwenChoice(args.model,local_files_only=args.local_files_only)
    case=json.loads((root/'examples/support_ticket.json').read_text())
    image=Image.open(root/'examples/support_ticket.png').convert('RGB')
    original_prepare=QwenChoice.prepare.__get__(model)
    fast_prepare=FastQwenChoice.prepare.__get__(model)

    # This audit is outside the timed request and compares the actual processor
    # tensors, including a new unreadable image to detect stale image reuse.
    checks=[]
    for sample in (image,Image.new('RGB',image.size,'white'),image):
        left=original_prepare(sample,case['question'],case['options'])
        right=fast_prepare(sample,case['question'],case['options'])
        same=all(torch.equal(left[0][key],right[0][key]) for key in left[0])
        assert same and left[1:3]==right[1:3]
        checks.append({'all_prepared_tensors_equal':same,
                       'pixel_sha256':hashlib.sha256(left[0]['pixel_values'].numpy().tobytes()).hexdigest()})
    assert checks[0]['pixel_sha256']!=checks[1]['pixel_sha256']
    assert checks[0]['pixel_sha256']==checks[2]['pixel_sha256']
    param_state=[(n,p.data_ptr(),p._version) for n,p in model.model.named_parameters()]
    setup_start=time.perf_counter()
    compile_calls={'graphs':0,'calls':0}
    if args.compile_language:
        from torch._dynamo.backends.registry import lookup_backend
        inductor=lookup_backend('inductor')
        def backend(gm,inputs):
            fn=inductor(gm,inputs)
            compile_calls['graphs']+=1
            def run(*a,**kw):
                compile_calls['calls']+=1
                return fn(*a,**kw)
            return run
        language=model.model.model.language_model
        language.forward=torch.compile(language.forward,backend=backend,dynamic=False,fullgraph=False)
    model.prepare=fast_prepare
    for _ in range(5): model.judge(image,case['question'],case['options'])
    setup_s=time.perf_counter()-setup_start
    results=[]
    log=(args.output/'records.jsonl').open('w')
    for index in range(30):
        order=['reference','fast'] if index%2==0 else ['fast','reference']
        for name in order:
            model.prepare=original_prepare if name=='reference' else fast_prepare
            row=model.judge(image,case['question'],case['options'])
            row.update(variant=name,index=index)
            results.append(row)
            log.write(json.dumps(row)+'\n'); log.flush()
    log.close()
    summary={
        'setup_and_five_warmup_s':setup_s,
        'compile_language':args.compile_language,'language_compilation':compile_calls,
        'input_checks':checks,
        'profiles':{name:{key:stats([r[key] for r in results if r['variant']==name])
                          for key in ('request_s','model_gpu_s','prepare_s')}
                    for name in ('reference','fast')},
        'all_choices_match':all(r['choice']==case['expected_choice'] for r in results),
        'max_paired_probability_diff':max(
            max(abs(a-b) for a,b in zip(
                next(r for r in results if r['index']==i and r['variant']=='reference')['probabilities'].values(),
                next(r for r in results if r['index']==i and r['variant']=='fast')['probabilities'].values()))
            for i in range(30)),
        'parameters_unchanged':param_state==[(n,p.data_ptr(),p._version) for n,p in model.model.named_parameters()],
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                         for name in ('qwen_choice.py','qwen_choice_fast.py','benchmark_fast.py')},
        'records_sha256':hashlib.sha256((args.output/'records.jsonl').read_bytes()).hexdigest(),
    }
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert summary['all_choices_match'] and summary['parameters_unchanged']
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
