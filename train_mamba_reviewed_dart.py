"""Small source-aligned DART correction set; 50 percent synthetic replay."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import train_mamba_multifacts as multi

base = multi.trainer.base
original_prepare = base.prepare
original_generate = base.generate
review_file = Path(__file__).with_name('dart_reviewed_v1.json')
test_rows = []


def prepare(args, tokenizer):
    curated = json.loads(review_file.read_text(encoding='utf-8'))
    legacy = original_prepare(SimpleNamespace(
        dart=args.dart, train_examples=2000, eval_examples=64), tokenizer)
    previous = {s: {r['key'] for r in rows} for s, rows in legacy.items()}
    raw = {}
    for split, expected in base.HASHES.items():
        path = args.dart / (split + '.json')
        if base.bench.digest(path) != expected:
            raise ValueError('DART source checksum differs: ' + split)
        raw[split] = json.loads(path.read_text(encoding='utf-8'))
    reserved = {base.key(r) for s in ('dev', 'test') for r in raw[s]}
    result = {}
    keys = {}
    for split in ('train', 'dev', 'test'):
        selected = []
        keys[split] = set()
        for index, text in curated[split]:
            source = raw[split][index]
            key = base.key(source)
            if key in keys[split] or key in previous[split]:
                raise ValueError('Duplicate or previously used graph: ' + str(index))
            if split == 'train' and key in reserved:
                raise ValueError('Training graph also appears in a reserved split')
            keys[split].add(key)
            prompt = 'Facts:\n' + '\n'.join(' | '.join(t) for t in source['tripleset']) + '\nText:'
            prefix = tokenizer.encode(prompt)
            answer = tokenizer.encode(' ' + text)
            if tokenizer.encode(prompt + ' ' + text) != prefix + answer:
                raise ValueError('Token boundary changed')
            ids = prefix + answer + [tokenizer.eos_token_id]
            if len(ids) > 128:
                raise ValueError('Example exceeds 128 tokens: ' + str(index))
            selected.append(dict(
                index=index, key=key, kind='reviewed-dart',
                review_sha256=base.bench.digest(review_file),
                driver_sha256=base.bench.digest(Path(__file__)),
                prompt=prompt, reference=text, ids=ids,
                labels=[-100] * len(prefix) + answer + [tokenizer.eos_token_id]))
        result[split] = selected
    if keys['dev'] & keys['test']:
        raise ValueError('Validation/test graph overlap')
    test_rows[:] = result['test']
    print('Reviewed DART:', {s: len(v) for s, v in result.items()}, flush=True)
    return result


def generate(model, tokenizer, rows, limit):
    if rows and rows[0].get('kind') == 'reviewed-dart':
        rows = test_rows
    return original_generate(model, tokenizer, rows, limit)


def main():
    base.prepare = prepare
    base.generate = generate
    defaults = {
        '--initial': 'runs/mamba130m_multifacts/best_adapter',
        '--run': 'runs/mamba130m_reviewed_dart',
        '--steps': '300',
    }
    for flag, value in defaults.items():
        if not any(a == flag or a.startswith(flag + '=') for a in sys.argv[1:]):
            sys.argv.extend([flag, value])
    multi.main()


if __name__ == '__main__':
    main()
