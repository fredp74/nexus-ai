"""Financial fact verbalization; real SEC reports are test-only."""
import argparse
from datetime import date
from decimal import Decimal
import hashlib
import itertools
import json
import math
import random
import signal
import time
from pathlib import Path
import train_mamba_reviewed_dart as replay
base = replay.base
FIELDS = ('revenue', 'operating_income', 'net_income')
TAGS = {'revenue': {'RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'}, 'operating_income': {'OperatingIncomeLoss'}, 'net_income': {'NetIncomeLoss'}}

def number(value):
    text = format(value, ',f')
    return text.rstrip('0').rstrip('.') if '.' in text else text

def row(tokenizer, entity, end, values, unit, style, seed, source):
    scale = Decimal(1000000 if unit == 'million' else 1000000000)
    amounts = {k: number(Decimal(values[k]) / scale) for k in FIELDS}
    r, o, n = (amounts[k] for k in FIELDS)
    facts = [(entity, 'YEAR_END', end), (entity, 'UNIT', 'USD ' + unit), (entity, 'REVENUE', r), (entity, 'OPERATING_INCOME', o), (entity, 'NET_INCOME', n)]
    random.Random(seed).shuffle(facts)
    prompt = 'Facts:\n' + '\n'.join((' | '.join(t) for t in facts)) + '\nText:'
    texts = [f'For the year ended {end}, {entity} reported revenue of USD {r} {unit}, operating income of USD {o} {unit}, and net income of USD {n} {unit}.', f'For its fiscal year ended {end}, {entity} reported USD {r} {unit} in revenue, USD {o} {unit} in operating income and USD {n} {unit} in net income.', f"{entity}'s results for the year ended {end} were revenue of USD {r} {unit}, operating income of USD {o} {unit} and net income of USD {n} {unit}.", f'Revenue at {entity} was USD {r} {unit} for the year ended {end}. Operating income was USD {o} {unit}, and net income was USD {n} {unit}.']
    text = texts[style]
    prefix = tokenizer.encode(prompt)
    answer = tokenizer.encode(' ' + text)
    if tokenizer.encode(prompt + ' ' + text) != prefix + answer:
        raise ValueError('Token boundary changed')
    ids = prefix + answer + [tokenizer.eos_token_id]
    if len(ids) > 192:
        raise ValueError('Example exceeds 192 tokens: ' + entity)
    return dict(index=entity + ':' + end + ':' + unit, entity=entity, end=end, unit=unit, values_usd={k: str(values[k]) for k in FIELDS}, source=source, prompt=prompt, reference=text, ids=ids, labels=[-100] * len(prefix) + answer + [tokenizer.eos_token_id])

def legacy_financial_data(tokenizer):
    first = 'Amber Arbor Aspen Birch Cedar Clear Copper Crest Delta Elm Fern Golden Granite Harbor Maple Silver'.split()
    last = 'Analytics Components Devices Dynamics Energy Engineering Foods Holdings Industries Logistics Materials Networks Products Research Systems Technologies'.split()
    names = [' '.join(x) for x in itertools.product(first, last)]
    random.Random(1911).shuffle(names)
    result = {s: [] for s in ('train', 'dev', 'test')}
    for i, entity in enumerate(names[:160]):
        split = 'train' if i < 128 else 'dev' if i < 144 else 'test'
        rng = random.Random(8100 + i)
        year = rng.randrange(2001, 2041)
        for variant in (0, 1):
            end = f'{year + variant}-' + rng.choice(['03-31', '06-30', '09-30', '12-31'])
            revenue = rng.randrange(1000, 900000) * 1000000
            millions = revenue // 1000000
            operating = rng.randrange(-millions // 5, millions // 2) * 1000000
            net = rng.randrange(-millions // 4, millions // 2) * 1000000
            values = dict(zip(FIELDS, (revenue, operating, net)))
            result[split].append(row(tokenizer, entity, end, values, rng.choice(['million', 'billion']), (i + variant) % 4, i * 2 + variant, {'kind': 'synthetic', 'not_real_company_data': True}))
    return result

def regression_rows(args, tokenizer):
    rows = []
    for symbol in args.companies:
        folder = args.reports_root / ('company_' + symbol + '2024')
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        raw_path = folder / 'companyfacts.json'
        if base.bench.digest(raw_path) != report['raw_sha256']:
            raise ValueError('SEC snapshot checksum differs: ' + symbol)
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        if str(raw['cik']).zfill(10) != str(report['cik']).zfill(10) or raw['entityName'] != report['entity']:
            raise ValueError('SEC entity mismatch')
        if report['currency'] != 'USD' or report['filed'] > report['as_of']:
            raise ValueError('Unsupported currency or filing cutoff')
        for label in ('current', 'previous'):
            period = report.get(label)
            if period is None:
                continue
            days = (date.fromisoformat(period['end']) - date.fromisoformat(period['start'])).days
            if not 330 <= days <= 400:
                raise ValueError('Expected annual financial facts')
            values = {}
            for field in FIELDS:
                fact = period['facts'][field]
                if fact['tag'] not in TAGS[field]:
                    raise ValueError('Unsupported SEC tag')
                expected = Decimal(fact['value_usd'])
                candidates = [Decimal(str(item['val'])) for item in raw['facts']['us-gaap'][fact['tag']]['units']['USD'] if item.get('start') == period['start'] and item.get('end') == period['end'] and (item.get('accn') == report['accession']) and (item.get('filed') == report['filed']) and (item.get('form') == '10-K')]
                if not expected.is_finite() or not candidates or any((v != expected for v in candidates)):
                    raise ValueError('Report value disagrees with raw SEC facts: ' + field)
                values[field] = str(expected)
            source = dict(kind='real-sec-test-only', raw_sha256=report['raw_sha256'], cik=report['cik'], accession=report['accession'], filed=report['filed'], filing_url=report['filing_url'], start=period['start'], end=period['end'])
            for unit in ('million', 'billion'):
                rows.append(row(tokenizer, report['entity'], period['end'], values, unit, 0, len(rows), source))
        print('Verified SEC report:', report['entity'], flush=True)
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('runs/_models/mamba130m_1e76775f'))
    p.add_argument('--initial', type=Path, default=Path('runs/nexus_finance_v1/best_adapter'))
    p.add_argument('--dart', type=Path, default=Path('data/dart_cache'))
    p.add_argument('--reports-root', type=Path, default=Path('.migration_archive/source_data/data'))
    p.add_argument('--companies', nargs='+', default=['msft', 'aapl', 'googl', 'meta'])
    p.add_argument('--run', type=Path, default=Path('runs/nexus_finance_v3'))
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--until', type=int)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    args.until = args.steps if args.until is None else args.until
    if min(args.steps, args.threads) < 1 or not 1 <= args.until <= args.steps:
        p.error('Positive steps/threads and 1 <= until <= steps required')
    base.torch.set_num_threads(args.threads)
    digest = base.bench.digest
    if digest(args.cache / 'model.safetensors') != base.bench.MODEL_HASH:
        raise ValueError('Official model checksum differs')
    tokenizer = base.AutoTokenizer.from_pretrained(str(args.cache), local_files_only=True)
    finance = financial_data(tokenizer)
    real = real_rows(args, tokenizer)
    synthetic = replay.multi.controlled(tokenizer)
    dart = replay.prepare(args, tokenizer)
    data = {}
    for split in ('train', 'dev', 'test'):
        retained = [r for family in ('birthplace', 'owner', 'price') for r in synthetic[split][family]]
        data[split] = dict(finance=finance[split], synthetic_replay=retained if split == 'train' else retained[:4] + retained[24:28] + retained[48:52], dart_replay=dart[split])
    data['test']['real_sec'] = real
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True).encode()
    identity = dict(script=digest(Path(__file__)), helper=digest(Path(base.__file__)), source=digest(args.initial / 'adapter_model.safetensors'), source_config=digest(args.initial / 'adapter_config.json'), data=hashlib.sha256(payload).hexdigest(), steps=args.steps, lr=2e-05, seed=42, accumulation=4, max_length=192, checker=digest(Path(verifier.__file__)))
    args.run.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run / 'last.pt'
    if not checkpoint.exists() and any(args.run.iterdir()):
        raise FileExistsError('Use an empty run directory')
    config = base.MambaConfig.from_pretrained(str(args.cache), local_files_only=True)
    config.use_mambapy, config.use_cache = (True, False)
    backbone = base.MambaForCausalLM.from_pretrained(str(args.cache), config=config, local_files_only=True, torch_dtype=base.torch.float32, use_safetensors=True)
    model = base.PeftModel.from_pretrained(backbone, str(args.initial), is_trainable=True)
    params = [v for v in model.parameters() if v.requires_grad]
    optimizer = base.torch.optim.AdamW(params, lr=2e-05, weight_decay=0.0)
    rng = random.Random(42)
    base.torch.manual_seed(42)
    step, best, history = (0, float('inf'), [])
    if checkpoint.exists():
        saved = base.torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Resume inputs or script changed')
        base.set_peft_model_state_dict(model, saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng'])
        base.torch.set_rng_state(saved['torch_rng'])
        step, best, history = (saved['step'], saved['best'], saved['history'])
    else:
        (args.run / 'data.json').write_bytes(payload)

    def snapshot():
        base.save(checkpoint, dict(identity=identity, step=step, best=best, history=history, adapter=base.get_peft_model_state_dict(model), optimizer=optimizer.state_dict(), rng=rng.getstate(), torch_rng=base.torch.get_rng_state()))

    def metrics(split):
        return {name: base.evaluate(model, rows) for name, rows in data[split].items()}

    def validate():
        nonlocal best
        values = metrics('dev')
        selected_rows = data['dev']['finance'][:16]
        texts = base.generate(model, tokenizer, selected_rows, 128)
        checks = [verifier.check(r['reference'], t['generated']) for r, t in zip(selected_rows, texts)]
        exact = sum((c['passed'] for c in checks))
        weighted = 0.5 * values['finance'] + 0.25 * values['synthetic_replay'] + 0.25 * values['dart_replay']
        score = 1.0 - exact / len(checks) + 0.001 * weighted
        initial = history[0] if history else None
        retained = initial is None or all((values[k] <= initial['nll'][k] * 1.1 + 0.02 for k in ('synthetic_replay', 'dart_replay')))
        minimum = max(12, initial['exact']) if initial else 0
        eligible = step == 0 or (retained and exact >= minimum)
        history.append(dict(step=step, nll=values, exact=exact, tested=len(checks), retention_ok=retained, eligible=eligible, selection=score, generations=texts, checks=checks))
        if eligible and score < best:
            best = score
            model.save_pretrained(args.run / 'best_adapter', safe_serialization=True)
            print('Selected candidate at step:', step, flush=True)
        print('Validation generated:', str(exact) + '/' + str(len(checks)), '| retention:', retained, '| eligible:', eligible, flush=True)
    stopped = [False]
    handler = signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    try:
        if not checkpoint.exists():
            validate()
            snapshot()
        started, start = (time.monotonic(), step)
        while step < args.until and (not stopped[0]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rows = rng.choices(data['train']['finance'], k=2)
            rows += [rng.choice(data['train']['synthetic_replay']), rng.choice(data['train']['dart_replay'])]
            logged = 0.0
            for item in rows:
                loss = base.loss_for(model, item) / 4
                if not base.torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                logged += loss.item()
            norm = base.torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not base.torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            warmup = min(40, max(1, args.steps // 10))
            progress = max(0.0, (step + 1 - warmup) / max(1, args.steps - warmup))
            lr = 2e-05 * min(1.0, (step + 1) / warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
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
        if step == args.steps and (not stopped[0]):
            model.load_adapter(str(args.run / 'best_adapter'), adapter_name='selected', is_trainable=False)
            model.set_adapter('selected')
            results = {}
            for mode in ('after', 'before'):
                if mode == 'before':
                    model.load_adapter(str(args.initial), adapter_name='before', is_trainable=False)
                    model.set_adapter('before')
                print('Evaluating:', mode, flush=True)
                texts = []
                for item in real:
                    texts += base.generate(model, tokenizer, [item], 96)
                    print(mode, len(texts), '/', len(real), flush=True)
                results[mode] = dict(nll=metrics('test'), real_generations=texts)
                base.write_json(args.run / 'result.json', dict(identity=identity, history=history, results=results))
            print('BEFORE NLL:', results['before']['nll'], flush=True)
            print('AFTER NLL:', results['after']['nll'], flush=True)
            final_scores = {}
            for mode, outcome in results.items():
                grouped = {}
                for source_row, generated in zip(real, outcome['real_generations']):
                    checked = verifier.check(source_row['reference'], generated['generated'])
                    group = source_row['source']['kind']
                    grouped.setdefault(group, []).append(dict(index=source_row['index'], **checked))
                final_scores[mode] = grouped
                for group, evaluated in grouped.items():
                    print(mode, group, sum((c['passed'] for c in evaluated)), '/', len(evaluated), flush=True)
            base.write_json(args.run / 'generated_scores.json', final_scores)
        if digest(args.initial / 'adapter_model.safetensors') != identity['source']:
            raise ValueError('Initial adapter changed')
        print('Saved:', args.run, '| step:', step, flush=True)
    finally:
        signal.signal(signal.SIGINT, handler)
import rescore_finance as verifier

def financial_data(tokenizer):
    result = {s: [] for s in ('train', 'dev', 'test')}
    first = 'Alder Alpine Azure Beacon Bloom Canyon Coral Dune Echo Falcon Frost Hazel Indigo Juniper Lagoon Laurel'.split()
    last = 'Advisory Aerospace Apparel Ceramics Chemicals Consulting Electronics Equipment Furniture Insurance Minerals Packaging Plastics Robotics Services Ventures'.split()
    names = [a + ' ' + b for a in first for b in last]
    random.Random(73191).shuffle(names)
    for i, name in enumerate(names[:160]):
        split = 'train' if i < 128 else 'dev' if i < 144 else 'test'
        entity = name + (' Inc.' if i % 2 else ' Corporation')
        rng = random.Random(63000 + i)
        year = rng.randrange(2001, 2040)
        closing = rng.choice(['03-31', '06-30', '09-30', '12-31'])
        for offset in (0, 1):
            revenue = rng.randrange(1000, 900000)
            operating = rng.randrange(-revenue // 5, revenue // 2)
            net = rng.randrange(-revenue // 4, revenue // 2)
            values = dict(zip(FIELDS, [v * 1000000 for v in (revenue, operating, net)]))
            for unit in ('million', 'billion'):
                item = row(tokenizer, entity, str(year + offset) + '-' + closing, values, unit, 0, i * 4 + offset, {'kind': 'synthetic-v3', 'not_real_company_data': True, 'split': split})
                result[split].append(item)
    sets = {split: {r['entity'] for r in rows} for split, rows in result.items()}
    assert not (sets['train'] & sets['dev'] or sets['train'] & sets['test'] or sets['dev'] & sets['test'])
    for rows in result.values():
        for item in rows:
            assert verifier.check(item['reference'], item['reference'])['passed']
    result['train'] += legacy_financial_data(tokenizer)['train']
    print('V3 synthetic examples:', {s: len(v) for s, v in result.items()}, flush=True)
    return result

def real_rows(args, tokenizer):
    heldout = financial_data(tokenizer)['test']
    return heldout + regression_rows(args, tokenizer)
if __name__ == '__main__':
    main()
