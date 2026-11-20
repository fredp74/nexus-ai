"""Metric-by-metric V5 from V3; V4 remains unchanged."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace
import train_nexus_finance_v3 as old
import nexus_metric as comment
import nexus_compare as compare
import rescore_finance as verifier
base = old.base

def pack(tokenizer, index, prompt, reference, kind):
    prefix = tokenizer.encode(prompt)
    answer = tokenizer.encode(' ' + reference)
    if tokenizer.encode(prompt + ' ' + reference) != prefix + answer:
        raise ValueError('Token boundary differs')
    ids = prefix + answer + [tokenizer.eos_token_id]
    if len(ids) > 384:
        raise ValueError('Example exceeds 384 tokens: ' + index)
    return dict(index=index, prompt=prompt, reference=reference, kind=kind, ids=ids, labels=[-100] * len(prefix) + answer + [tokenizer.eos_token_id])

def comparison_data(tokenizer):
    return comment.synthetic_data(tokenizer, pack)

def checked(row, generated):
    if row.get('kind') in ('synthetic-comparison', 'sec-comparison-regression'):
        return comment.conforms(generated, row['reference'])
    return verifier.check(row['reference'], generated)['passed']

def generated_score(model, tokenizer, rows):
    limit = 48 if rows[0].get('kind') in ('synthetic-comparison', 'sec-comparison-regression') else 128
    texts = base.generate(model, tokenizer, rows, limit)
    checks = [checked(row, text['generated']) for row, text in zip(rows, texts)]
    return dict(exact=sum(checks), tested=len(rows), checks=checks, generations=texts)

def real_tests(tokenizer):
    return comment.real_tests(tokenizer, pack, old)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--until', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.until = args.steps if args.until is None else args.until
    if not 1 <= args.until <= args.steps:
        parser.error('Require 1 <= until <= steps')
    base.torch.set_num_threads(8)
    cache = Path('runs/_models/mamba130m_1e76775f')
    initial = Path('runs/nexus_finance_v3/best_adapter')
    run = Path('runs/nexus_finance_v5')
    tokenizer = base.AutoTokenizer.from_pretrained(str(cache), local_files_only=True)
    comparisons = comparison_data(tokenizer)
    financial = old.financial_data(tokenizer)
    synthetic = old.replay.multi.controlled(tokenizer)
    dart = old.replay.prepare(SimpleNamespace(dart=Path('data/dart_cache')), tokenizer)
    data = {}
    for split in ('train', 'dev', 'test'):
        retained = [row for family in ('birthplace', 'owner', 'price') for row in synthetic[split][family]]
        data[split] = dict(comparison=comparisons[split], finance=financial[split], dart=dart[split], relation=retained[:4] + retained[24:28] + retained[48:52])
    print('Comparison examples:', {split: len(rows) for split, rows in comparisons.items()}, flush=True)
    print('Maximum comparison tokens:', max((len(row['ids']) for rows in comparisons.values() for row in rows)), flush=True)
    if args.prepare_only:
        return
    if base.bench.digest(cache / 'model.safetensors') != base.bench.MODEL_HASH:
        raise ValueError('Official backbone checksum differs')
    payload = json.dumps(data, sort_keys=True).encode()
    identity = dict(scripts={Path(module.__file__).name: base.bench.digest(Path(module.__file__)) for module in (old, comment, compare, verifier, base)}, script=base.bench.digest(Path(__file__)), data=hashlib.sha256(payload).hexdigest(), steps=args.steps, source={name: base.bench.digest(initial / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}, lr=5e-05, seed=20261120)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / 'last.pt'
    if not checkpoint.exists() and any(run.iterdir()):
        raise FileExistsError('Use an empty V4 run directory')
    config = base.MambaConfig.from_pretrained(str(cache), local_files_only=True)
    config.use_mambapy = True
    config.use_cache = False
    backbone = base.MambaForCausalLM.from_pretrained(str(cache), config=config, local_files_only=True, torch_dtype=base.torch.float32, use_safetensors=True)
    model = base.PeftModel.from_pretrained(backbone, str(initial), is_trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = base.torch.optim.AdamW(params, lr=5e-05, weight_decay=0.0)
    rng = random.Random(20261120)
    base.torch.manual_seed(20261120)
    step, best, history = (0, float('inf'), [])
    if checkpoint.exists():
        saved = base.torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Resume inputs changed')
        base.set_peft_model_state_dict(model, saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng'])
        base.torch.set_rng_state(saved['torch_rng'])
        step = saved['step']
        best = saved['best']
        history = saved['history']
    else:
        (run / 'data.json').write_bytes(payload)

    def snapshot():
        base.save(checkpoint, dict(identity=identity, step=step, best=best, history=history, adapter=base.get_peft_model_state_dict(model), optimizer=optimizer.state_dict(), rng=rng.getstate(), torch_rng=base.torch.get_rng_state()))

    def validate():
        nonlocal best
        nll = {key: base.evaluate(model, rows) for key, rows in data['dev'].items()}
        a = generated_score(model, tokenizer, data['dev']['comparison'][:20])
        b = generated_score(model, tokenizer, data['dev']['finance'][:16])
        first = history[0] if history else None
        retained = first is None or (b['exact'] >= first['finance']['exact'] and all((nll[key] <= first['nll'][key] * 1.1 + 0.02 for key in ('dart', 'relation'))))
        eligible = step == 0 or (retained and a['exact'] >= max(18, first['comparison']['exact']))
        score = 1 - a['exact'] / 20 + 0.0001 * nll['comparison']
        history.append(dict(step=step, nll=nll, comparison=a, finance=b, retained=retained, eligible=eligible))
        if eligible and score < best:
            best = score
            model.save_pretrained(run / 'best_adapter', safe_serialization=True)
            base.write_json(run / 'selection.json', dict(step=step, comparison=a['exact'], finance=b['exact'], retained=retained))
        print('Validation', step, '| metrics', a['exact'], '/20', '| summaries', b['exact'], '/16', '| retention', retained, '| eligible', eligible, flush=True)
    stop = [False]
    handler = signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    try:
        if not checkpoint.exists():
            validate()
            snapshot()
        started, start = (time.monotonic(), step)
        while step < args.until and (not stop[0]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rows = rng.choices(data['train']['comparison'], k=2) + [rng.choice(data['train']['finance']), rng.choice(data['train']['dart'])]
            logged = 0.0
            for row in rows:
                loss = base.loss_for(model, row) / 4
                if not base.torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                logged += loss.item()
            norm = base.torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not base.torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            warmup = min(20, max(1, args.steps // 10))
            progress = max(0.0, (step + 1 - warmup) / max(1, args.steps - warmup))
            lr = 5e-05 * min(1.0, (step + 1) / warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            step += 1
            if step % 10 == 0:
                eta = (time.monotonic() - started) / (step - start) * (args.until - step) / 60
                print('Step', step, '/', args.steps, '| loss', round(logged, 4), '| training ETA min', round(eta, 1), flush=True)
            if step % 50 == 0 or step == args.until:
                validate()
            if step % 10 == 0 or step == args.until or stop[0]:
                snapshot()
        snapshot()
        if step == args.steps and (not stop[0]):
            tests = comparisons['test'] + real_tests(tokenizer)
            result = dict(identity=identity, history=history, results={})
            for mode, path in (('after', run / 'best_adapter'), ('before', initial)):
                model.load_adapter(str(path), adapter_name=mode, is_trainable=False)
                model.set_adapter(mode)
                evaluated = []
                for row in tests:
                    text = base.generate(model, tokenizer, [row], 48 if row.get('kind') in ('synthetic-comparison', 'sec-comparison-regression') else 128)[0]
                    evaluated.append(dict(**text, kind=row['kind'], source=row.get('source'), passed=checked(row, text['generated'])))
                    result['results'][mode] = evaluated
                    base.write_json(run / 'result.json', result)
                    print(mode, len(evaluated), '/', len(tests), flush=True)
                for kind in sorted({row['kind'] for row in evaluated}):
                    group = [row for row in evaluated if row['kind'] == kind]
                    print(mode, kind, sum((row['passed'] for row in group)), '/', len(group), flush=True)
        if any((base.bench.digest(initial / name) != value for name, value in identity['source'].items())):
            raise ValueError('Source V3 changed')
        print('Selected:', json.loads((run / 'selection.json').read_text(encoding='utf-8')), flush=True)
        print('Saved:', run, '| step:', step, flush=True)
    finally:
        signal.signal(signal.SIGINT, handler)
if __name__ == '__main__':
    main()
