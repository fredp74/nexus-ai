"""Evaluate V4 last checkpoint without changing adapters or selection."""
import hashlib
import json
from pathlib import Path

import train_nexus_finance_v4 as trainer

base = trainer.base


def fingerprint(path):
    return base.bench.digest(path)


def main():
    base.torch.set_num_threads(8)
    run = Path("runs/nexus_finance_v4")
    checkpoint = run / "last.pt"
    checkpoint_hash = fingerprint(checkpoint)
    saved = base.torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    identity = saved["identity"]

    if fingerprint(Path(trainer.__file__)) != identity["script"]:
        raise ValueError(
            "Training script changed since checkpoint creation"
        )
    for module in (
        trainer.old, trainer.comment,
        trainer.compare, trainer.verifier, base,
    ):
        path = Path(module.__file__)
        if fingerprint(path) != identity["scripts"][path.name]:
            raise ValueError(
                "Training dependency changed: " + path.name
            )

    data_path = run / "data.json"
    if fingerprint(data_path) != identity["data"]:
        raise ValueError("Frozen dataset checksum differs")

    initial = Path("runs/nexus_finance_v3/best_adapter")
    for name, expected in identity["source"].items():
        if fingerprint(initial / name) != expected:
            raise ValueError("Source V3 differs")

    selection_hash = fingerprint(run / "selection.json")
    cache = Path("runs/_models/mamba130m_1e76775f")
    if (
        fingerprint(cache / "model.safetensors")
        != base.bench.MODEL_HASH
    ):
        raise ValueError("Official backbone checksum differs")

    tokenizer = base.AutoTokenizer.from_pretrained(
        str(cache), local_files_only=True
    )
    data = json.loads(data_path.read_text(encoding="utf-8"))
    tests = (
        data["test"]["comparison"]
        + trainer.real_tests(tokenizer)
    )
    keys = [(row["kind"], row["index"]) for row in tests]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate test identifiers")

    audit_identity = dict(
        checkpoint_sha256=checkpoint_hash,
        step=saved["step"],
        script_sha256=fingerprint(Path(__file__)),
        data_sha256=identity["data"],
        test_sha256=hashlib.sha256(
            json.dumps(tests, sort_keys=True).encode()
        ).hexdigest(),
        tokens=128,
        selection_sha256=selection_hash,
    )
    output = run / (
        "audit_last_step" + str(saved["step"]) + ".json"
    )
    result = dict(
        identity=audit_identity,
        rows=[],
        role="Diagnostic only; no candidate promotion",
    )

    if output.exists():
        result = json.loads(
            output.read_text(encoding="utf-8")
        )
        if result["identity"] != audit_identity:
            raise ValueError("Audit inputs changed")

    done = {
        (row["kind"], row["index"])
        for row in result["rows"]
    }
    print(
        "Evaluating trained last checkpoint, step:",
        saved["step"], flush=True,
    )

    if len(done) != len(tests):
        config = base.MambaConfig.from_pretrained(
            str(cache), local_files_only=True
        )
        config.use_mambapy = True
        config.use_cache = False
        backbone = base.MambaForCausalLM.from_pretrained(
            str(cache), config=config,
            local_files_only=True,
            torch_dtype=base.torch.float32,
            use_safetensors=True,
        )
        model = base.PeftModel.from_pretrained(
            backbone, str(initial), is_trainable=False
        )
        base.set_peft_model_state_dict(
            model, saved["adapter"]
        )

        loaded = base.get_peft_model_state_dict(model)
        if (
            set(loaded) != set(saved["adapter"])
            or any(
                not loaded[key].equal(value)
                for key, value in saved["adapter"].items()
            )
        ):
            raise ValueError(
                "Checkpoint adapter was not loaded exactly"
            )
        model.eval().requires_grad_(False)

        for row in tests:
            if (row["kind"], row["index"]) in done:
                continue
            generated = base.generate(
                model, tokenizer, [row], 128
            )[0]
            result["rows"].append(dict(
                **generated,
                kind=row["kind"],
                source=row.get("source"),
                passed=trainer.checked(
                    row, generated["generated"]
                ),
            ))
            base.write_json(output, result)
            print(
                "Evaluated:", len(result["rows"]),
                "/", len(tests), flush=True,
            )

    for kind in sorted({
        row["kind"] for row in result["rows"]
    }):
        group = [
            row for row in result["rows"]
            if row["kind"] == kind
        ]
        print(
            kind,
            sum(row["passed"] for row in group),
            "/", len(group), flush=True,
        )

    for row in result["rows"]:
        if row["kind"] == "sec-comparison-regression":
            print(
                "\nCASE:", row["index"],
                "| passed:", row["passed"],
            )
            print("REFERENCE:", row["reference"])
            print("GENERATED:", row["generated"])

    if (
        fingerprint(checkpoint) != checkpoint_hash
        or fingerprint(run / "selection.json") != selection_hash
    ):
        raise ValueError(
            "Source checkpoint or selection changed"
        )
    for name, expected in identity["source"].items():
        if fingerprint(initial / name) != expected:
            raise ValueError("Source V3 changed")

    print("\nDiagnostic only. V3 and candidate selection unchanged.")
    print("Saved:", output)


if __name__ == "__main__":
    main()
