"""Build an isolated V3 trainer from the existing V1 implementation."""
import ast
from pathlib import Path

EXTRA = '''
def financial_data(tokenizer):
    result = {s: [] for s in ('train', 'dev', 'test')}
    first = (
        'Alder Alpine Azure Beacon Bloom Canyon Coral Dune Echo '
        'Falcon Frost Hazel Indigo Juniper Lagoon Laurel'
    ).split()
    last = (
        'Advisory Aerospace Apparel Ceramics Chemicals Consulting '
        'Electronics Equipment Furniture Insurance Minerals Packaging '
        'Plastics Robotics Services Ventures'
    ).split()
    names = [a + ' ' + b for a in first for b in last]
    random.Random(73191).shuffle(names)

    for i, name in enumerate(names[:160]):
        split = (
            'train' if i < 128
            else ('dev' if i < 144 else 'test')
        )
        entity = name + (' Inc.' if i % 2 else ' Corporation')
        rng = random.Random(63000 + i)
        year = rng.randrange(2001, 2040)
        closing = rng.choice([
            '03-31', '06-30', '09-30', '12-31'
        ])

        for offset in (0, 1):
            revenue = rng.randrange(1000, 900000)
            operating = rng.randrange(
                -revenue // 5, revenue // 2
            )
            net = rng.randrange(-revenue // 4, revenue // 2)
            values = dict(zip(
                FIELDS,
                [v * 1000000 for v in (revenue, operating, net)]
            ))

            for unit in ('million', 'billion'):
                item = row(
                    tokenizer, entity,
                    str(year + offset) + '-' + closing,
                    values, unit, 0, i * 4 + offset,
                    {
                        'kind': 'synthetic-v3',
                        'not_real_company_data': True,
                        'split': split,
                    },
                )
                result[split].append(item)

    sets = {
        split: {r['entity'] for r in rows}
        for split, rows in result.items()
    }
    assert not (
        sets['train'] & sets['dev']
        or sets['train'] & sets['test']
        or sets['dev'] & sets['test']
    )
    for rows in result.values():
        for item in rows:
            assert verifier.check(
                item['reference'], item['reference']
            )['passed']

    result['train'] += legacy_financial_data(tokenizer)['train']
    print(
        'V3 synthetic examples:',
        {s: len(v) for s, v in result.items()},
        flush=True,
    )
    return result


def real_rows(args, tokenizer):
    heldout = financial_data(tokenizer)['test']
    return heldout + regression_rows(args, tokenizer)
'''

VALIDATE = '''
def validate():
    nonlocal best
    values = metrics('dev')
    selected_rows = data['dev']['finance'][:16]
    texts = base.generate(model, tokenizer, selected_rows, 128)
    checks = [
        verifier.check(r['reference'], t['generated'])
        for r, t in zip(selected_rows, texts)
    ]
    exact = sum(c['passed'] for c in checks)
    weighted = (
        .5 * values['finance']
        + .25 * values['synthetic_replay']
        + .25 * values['dart_replay']
    )
    score = 1. - exact / len(checks) + .001 * weighted
    initial = history[0] if history else None
    retained = initial is None or all(
        values[k] <= initial['nll'][k] * 1.10 + .02
        for k in ('synthetic_replay', 'dart_replay')
    )
    minimum = max(12, initial['exact']) if initial else 0
    eligible = step == 0 or (retained and exact >= minimum)

    history.append(dict(
        step=step, nll=values, exact=exact, tested=len(checks),
        retention_ok=retained, eligible=eligible,
        selection=score, generations=texts, checks=checks,
    ))
    if eligible and score < best:
        best = score
        model.save_pretrained(
            args.run / 'best_adapter',
            safe_serialization=True,
        )
        print('Selected candidate at step:', step, flush=True)

    print(
        'Validation generated:',
        str(exact) + '/' + str(len(checks)),
        '| retention:', retained,
        '| eligible:', eligible,
        flush=True,
    )
'''

FINAL_SCORES = '''
final_scores = {}
for mode, outcome in results.items():
    grouped = {}
    for source_row, generated in zip(
        real, outcome['real_generations']
    ):
        checked = verifier.check(
            source_row['reference'], generated['generated']
        )
        group = source_row['source']['kind']
        grouped.setdefault(group, []).append(
            dict(index=source_row['index'], **checked)
        )
    final_scores[mode] = grouped
    for group, evaluated in grouped.items():
        print(
            mode, group,
            sum(c['passed'] for c in evaluated),
            '/', len(evaluated), flush=True,
        )
base.write_json(
    args.run / 'generated_scores.json', final_scores
)
'''


def main():
    source = Path('train_nexus_finance.py')
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == 'financial_data':
                node.name = 'legacy_financial_data'
            elif node.name == 'real_rows':
                node.name = 'regression_rows'

    counter = {
        'validate': 0, 'initial': 0, 'run': 0, 'steps': 0
    }

    class Rewrite(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'validate':
                counter['validate'] += 1
                return ast.parse(VALIDATE).body[0]
            return self.generic_visit(node)

        def visit_Constant(self, node):
            changes = {
                'runs/mamba130m_reviewed_dart/best_adapter':
                    'runs/nexus_finance_v1/best_adapter',
                'runs/nexus_finance_v1':
                    'runs/nexus_finance_v3',
            }
            if isinstance(node.value, str) and node.value in changes:
                key = (
                    'initial'
                    if 'reviewed_dart' in node.value else 'run'
                )
                counter[key] += 1
                return ast.copy_location(
                    ast.Constant(changes[node.value]), node
                )
            if isinstance(node.value, float) and node.value == .00005:
                return ast.copy_location(
                    ast.Constant(.00002), node
                )
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if any(
                isinstance(t, ast.Name) and t.id == 'identity'
                for t in node.targets
            ):
                expression = ast.parse(
                    "digest(Path(verifier.__file__))",
                    mode='eval',
                ).body
                node.value.keywords.append(
                    ast.keyword(arg='checker', value=expression)
                )
            return node

        def visit_For(self, node):
            if ast.unparse(node.iter) == (
                "results['after']['real_generations']"
            ):
                return ast.parse(FINAL_SCORES).body
            return self.generic_visit(node)

        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == '--steps'
            ):
                for keyword in node.keywords:
                    if keyword.arg == 'default':
                        keyword.value = ast.Constant(200)
                        counter['steps'] += 1
            return node

    tree = Rewrite().visit(tree)
    assert counter == {
        'validate': 1, 'initial': 1, 'run': 1, 'steps': 1
    }, counter

    tree.body[-1:-1] = ast.parse(
        'import rescore_finance as verifier\n' + EXTRA
    ).body
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + '\n'
    compile(text, 'train_nexus_finance_v3.py', 'exec')

    target = Path('train_nexus_finance_v3.py')
    if target.exists() and target.read_text(encoding='utf-8') != text:
        raise FileExistsError(
            'Existing V3 script differs; not overwritten'
        )
    if Path('runs/nexus_finance_v3').exists():
        raise FileExistsError(
            'V3 run already exists; resume its unchanged trainer'
        )
    target.write_text(text, encoding='utf-8')
    print('Created:', target)
    print('Source V1 trainer and adapter unchanged.')


if __name__ == '__main__':
    main()
