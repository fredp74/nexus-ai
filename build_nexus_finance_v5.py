"""Build an isolated metric-by-metric V5 trainer from V4 code."""
import ast
from pathlib import Path

FUNCTIONS = {
    "comparison_data": """
def comparison_data(tokenizer):
    return comment.synthetic_data(tokenizer, pack)
""",
    "real_tests": """
def real_tests(tokenizer):
    return comment.real_tests(tokenizer, pack, old)
""",
    "generated_score": """
def generated_score(model, tokenizer, rows):
    limit = (
        48 if rows[0].get('kind') in (
            'synthetic-comparison', 'sec-comparison-regression'
        ) else 128
    )
    texts = base.generate(model, tokenizer, rows, limit)
    checks = [
        checked(row, text['generated'])
        for row, text in zip(rows, texts)
    ]
    return dict(
        exact=sum(checks), tested=len(rows),
        checks=checks, generations=texts
    )
""",
    "validate": """
def validate():
    nonlocal best
    nll = {
        key: base.evaluate(model, rows)
        for key, rows in data['dev'].items()
    }
    a = generated_score(
        model, tokenizer, data['dev']['comparison'][:20]
    )
    b = generated_score(
        model, tokenizer, data['dev']['finance'][:16]
    )
    first = history[0] if history else None
    retained = first is None or (
        b['exact'] >= first['finance']['exact']
        and all(
            nll[key] <= first['nll'][key] * 1.10 + .02
            for key in ('dart', 'relation')
        )
    )
    eligible = step == 0 or (
        retained
        and a['exact'] >= max(18, first['comparison']['exact'])
    )
    score = 1 - a['exact'] / 20 + .0001 * nll['comparison']
    history.append(dict(
        step=step, nll=nll,
        comparison=a, finance=b,
        retained=retained, eligible=eligible
    ))
    if eligible and score < best:
        best = score
        model.save_pretrained(
            run / 'best_adapter', safe_serialization=True
        )
        base.write_json(
            run / 'selection.json',
            dict(
                step=step,
                comparison=a['exact'],
                finance=b['exact'],
                retained=retained
            )
        )
    print(
        'Validation', step,
        '| metrics', a['exact'], '/20',
        '| summaries', b['exact'], '/16',
        '| retention', retained,
        '| eligible', eligible,
        flush=True
    )
""",
}


def main():
    if Path("runs/nexus_finance_v5").exists():
        raise FileExistsError(
            "V5 run exists; resume its unchanged trainer"
        )
    if not Path("nexus_metric.py").is_file():
        raise FileNotFoundError("nexus_metric.py")

    tree = ast.parse(
        Path("train_nexus_finance_v4.py").read_text(
            encoding="utf-8-sig"
        )
    )
    counts = {key: 0 for key in FUNCTIONS}
    counts.update(imports=0, run=0, generation=0)

    class Rewrite(ast.NodeTransformer):
        def visit_Import(self, node):
            for alias in node.names:
                if (
                    alias.name == "nexus_comment"
                    and alias.asname == "comment"
                ):
                    alias.name = "nexus_metric"
                    counts["imports"] += 1
            return node

        def visit_Constant(self, node):
            if node.value == "runs/nexus_finance_v4":
                counts["run"] += 1
                return ast.copy_location(
                    ast.Constant("runs/nexus_finance_v5"),
                    node,
                )
            return node

        def visit_FunctionDef(self, node):
            if node.name in FUNCTIONS:
                counts[node.name] += 1
                return ast.parse(FUNCTIONS[node.name]).body[0]
            return self.generic_visit(node)

        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == "base.generate":
                assert len(node.args) == 4
                node.args[3] = ast.parse(
                    "48 if row.get('kind') in "
                    "('synthetic-comparison', "
                    "'sec-comparison-regression') else 128",
                    mode="eval",
                ).body
                counts["generation"] += 1
            return node

    tree = Rewrite().visit(tree)
    assert all(
        value == 1 for value in counts.values()
    ), counts

    if (
        isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    ):
        tree.body[0].value = ast.Constant(
            "Metric-by-metric V5 from V3; V4 remains unchanged."
        )

    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + "\n"
    compile(text, "train_nexus_finance_v5.py", "exec")

    target = Path("train_nexus_finance_v5.py")
    if (
        target.exists()
        and target.read_text(encoding="utf-8") != text
    ):
        raise FileExistsError("Existing V5 trainer differs")

    target.write_text(text, encoding="utf-8")
    print("Created:", target)
    print("Source V3 adapter and V4 experiment unchanged.")


if __name__ == "__main__":
    main()
