"""Continue in/out LoRA with controlled relations and 50 percent DART replay."""
import argparse
import hashlib
import itertools
import json
import math
import random
import signal
import time
from pathlib import Path

import train_mamba130m_lora as base


def controlled(tokenizer):
    first = 'Alina Cora Felix Hugo Ines Jonas Kira Leon Mara Nils Orla Pavel Rina Silas Tessa Vera'.split()
    last = 'Costa Duarte Fischer Keller Lind Meyer Novak Olsen Petrov Rossi Silva Vogel Weber Winter Young Zimmer'.split()
    adjectives = 'Amber Azure Bright Calm Clear Coral Distant Ivory Lunar Noble Pale Royal Scarlet Spring Wild Winter'.split()
    nouns = 'Anchor Beacon Bridge Compass Crown Dune Fern Gate Lake Leaf Moon Pearl Rock Sail Shore Willow'.split()
    cities = 'Alicante Antwerp Avignon Basel Bern Bonn Cork Delft Dijon Essen Gijon Graz Linz Metz Namur Parma'.split()
    pools = {
        'birthplace': [' '.join(x) for x in itertools.product(first, last)],
        'owner': ['SS ' + ' '.join(x) for x in itertools.product(adjectives, nouns)],
        'price': ['The ' + ' '.join(x) for x in itertools.product(adjectives, nouns)],
    }
    result = {s: {f: [] for f in pools} for s in ('train', 'dev', 'test')}
    rng = random.Random(1801)
    for family, names in pools.items():
        rng.shuffle(names)
        for i, name in enumerate(names[:152]):
            split = 'train' if i < 128 else ('dev' if i < 140 else 'test')
            values = rng.sample(cities, 2)
            owners = rng.sample(adjectives, 2)
            engine = rng.choice(nouns) + ' Engineering'
            prices = rng.sample(range(10, 100, 5), 2)
            for variant in (0, 1):
                if family == 'birthplace':
                    facts = [(name, 'BIRTH_PLACE', values[variant])]
                    text = name + ' was born in ' + values[variant] + '.'
                elif family == 'owner':
                    owner = owners[variant] + ' Shipping'
                    facts = [(name, 'OWNER', owner), (name, 'POWER_TYPE', engine)]
                    text = name + ' is owned by ' + owner + ' and powered by ' + engine + '.'
                    if i % 2:
                        facts.reverse()
                else:
                    value = ('less than ' if variant == 0 else 'more than ') + '\u00a3' + str(prices[variant])
                    facts = [(name, 'eatType', 'restaurant'), (name, 'priceRange', value)]
                    text = name + ' is a restaurant priced at ' + value + '.'
                prompt = 'Facts:\n' + '\n'.join(' | '.join(t) for t in facts) + '\nText:'
                prefix = tokenizer.encode(prompt)
                answer = tokenizer.encode(' ' + text)
                if tokenizer.encode(prompt + ' ' + text) != prefix + answer:
                    raise ValueError('Token boundary changed')
                ids = prefix + answer + [tokenizer.eos_token_id]
                if len(ids) > 128:
                    raise ValueError('Example exceeds 128 tokens')
                result[split][family].append(dict(
                    index=name + ':' + str(variant), entity=name, prompt=prompt,
                    reference=text, ids=ids,
                    labels=[-100] * len(prefix) + answer + [tokenizer.eos_token_id]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('runs/_models/mamba130m_1e76775f'))
    parser.add_argument('--initial', type=Path, default=Path('runs/mamba130m_inout_sft/inout/best_adapter'))
    parser.add_argument('--dart', type=Path, default=Path('data/dart_cache'))
    parser.add_argument('--run', type=Path, default=Path('runs/mamba130m_relations_replay'))
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--until', type=int)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    args.until = args.steps if args.until is None else args.until
    if min(args.steps, args.threads) < 1 or not 1 <= args.until <= args.steps:
        parser.error('Positive steps/threads and 1 <= until <= steps required')
    base.torch.set_num_threads(args.threads)
    digest = base.bench.digest
    if digest(args.cache / 'model.safetensors') != base.bench.MODEL_HASH:
        raise ValueError('Official weights differ')
    tokenizer = base.AutoTokenizer.from_pretrained(str(args.cache), local_files_only=True)
    data = controlled(tokenizer)
    args.train_examples, args.eval_examples = 2000, 32
    dart = base.prepare(args, tokenizer)
    for split in data:
        data[split]['dart'] = dart[split]
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True).encode()
    identity = dict(script=digest(Path(__file__)), helper=digest(Path(base.__file__)),
        source=digest(args.initial / 'adapter_model.safetensors'),
        source_config=digest(args.initial / 'adapter_config.json'),
        data=hashlib.sha256(payload).hexdigest(), steps=args.steps,
        lr=.0001, seed=42, batch=4, dart_fraction=.5)
    args.run.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run / 'last.pt'
    if not checkpoint.exists() and any(args.run.iterdir()):
        raise FileExistsError('Use an empty run directory')
    config = base.MambaConfig.from_pretrained(str(args.cache), local_files_only=True)
    config.use_mambapy, config.use_cache = True, False
    backbone = base.MambaForCausalLM.from_pretrained(str(args.cache), config=config,
        local_files_only=True, torch_dtype=base.torch.float32, use_safetensors=True)
    model = base.PeftModel.from_pretrained(backbone, str(args.initial), is_trainable=True)
    if set(model.peft_config['default'].target_modules) != {'in_proj', 'out_proj'}:
        raise ValueError('Expected in_proj + out_proj source adapter')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = base.torch.optim.AdamW(params, lr=.0001, weight_decay=0.)
    rng = random.Random(42)
    base.torch.manual_seed(42)
    step, best, history = 0, float('inf'), []
    if checkpoint.exists():
        state = base.torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Resume inputs or settings changed')
        base.set_peft_model_state_dict(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        rng.setstate(state['rng'])
        base.torch.set_rng_state(state['torch_rng'])
        step, best, history = state['step'], state['best'], state['history']
    else:
        (args.run / 'data.json').write_bytes(payload)

    def snapshot():
        base.save(checkpoint, dict(identity=identity, step=step, best=best, history=history,
            adapter=base.get_peft_model_state_dict(model), optimizer=optimizer.state_dict(),
            rng=rng.getstate(), torch_rng=base.torch.get_rng_state()))

    def metrics(split):
        return {f: base.evaluate(model, rows) for f, rows in data[split].items()}

    def validate():
        nonlocal best
        values = metrics('dev')
        score = sum(values.values()) / len(values)
        history.append(dict(step=step, nll=values, macro_nll=score))
        if score < best:
            best = score
            model.save_pretrained(args.run / 'best_adapter', safe_serialization=True)
        print('Validation:', values, '| macro:', round(score, 4), flush=True)

    stopped = [False]
    previous = signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    try:
        if not checkpoint.exists():
            validate()
            snapshot()
        started, start = time.monotonic(), step
        families = ['birthplace', 'owner', 'price']
        while step < args.until and not stopped[0]:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rows = [rng.choice(data['train'][rng.choice(families)]) for _ in range(2)]
            rows += rng.choices(data['train']['dart'], k=2)
            logged = 0.
            for row in rows:
                loss = base.loss_for(model, row) / 4
                if not base.torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                logged += loss.item()
            norm = base.torch.nn.utils.clip_grad_norm_(params, 1.)
            if not base.torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            warmup = min(50, max(1, args.steps // 10))
            progress = max(0., (step + 1 - warmup) / max(1, args.steps - warmup))
            lr = .0001 * min(1., (step + 1) / warmup) * (.1 + .45 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            step += 1
            if step % 10 == 0:
                eta = (time.monotonic() - started) / (step - start) * (args.until - step) / 60
                print('Step', step, '/', args.steps, '| loss:', round(logged, 4), '| ETA min:', round(eta, 1), flush=True)
            if step % 50 == 0 or step == args.until:
                validate()
            if step % 10 == 0 or step == args.until or stopped[0]:
                snapshot()
        snapshot()
        if step == args.steps and not stopped[0]:
            model.load_adapter(str(args.run / 'best_adapter'), adapter_name='selected', is_trainable=False)
            model.set_adapter('selected')
            result = dict(identity=identity, history=history, test_nll=metrics('test'), generations={})
            for family, rows in data['test'].items():
                result['generations'][family] = base.generate(model, tokenizer, rows[:4], 64)
            model.load_adapter(str(args.initial), adapter_name='before', is_trainable=False)
            model.set_adapter('before')
            result['before_test_nll'] = metrics('test')
            result['before_generations'] = {
                f: base.generate(model, tokenizer, rows[:4], 64)
                for f, rows in data['test'].items()}
            base.write_json(args.run / 'result.json', result)
            print('BEFORE TEST NLL:', result['before_test_nll'], flush=True)
            print('TEST NLL:', result['test_nll'], flush=True)
            for family, rows in result['generations'].items():
                for row in rows:
                    print(family, '|', row['reference'], '->', row['generated'], flush=True)
        if digest(args.initial / 'adapter_model.safetensors') != identity['source']:
            raise ValueError('Source adapter changed')
        print('Saved:', args.run, '| step:', step, flush=True)
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == '__main__':
    main()
