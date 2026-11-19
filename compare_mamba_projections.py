"""Compare out_proj SFT with an expanded in_proj + out_proj adapter."""
import argparse
import hashlib
import json
import signal
from pathlib import Path

import compare_mamba_faithfulness as exp
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import load_peft_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path,
        default=Path("runs/_models/mamba130m_1e76775f"))
    parser.add_argument("--source", type=Path,
        default=Path("runs/mamba130m_dart_lora/best_adapter"))
    parser.add_argument("--baseline", type=Path,
        default=Path("runs/mamba130m_faithfulness_training"))
    parser.add_argument("--run", type=Path,
        default=Path("runs/mamba130m_inout_sft"))
    parser.add_argument("--until", type=int)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    exp.torch.set_num_threads(args.threads)

    report = json.loads(
        (args.baseline / "comparison.json").read_text(encoding="utf-8")
    )
    previous = report["identity"]
    args.updates = previous["updates"]
    args.until = args.updates if args.until is None else args.until
    if not 1 <= args.until <= args.updates:
        parser.error("until must be between 1 and the original update budget")

    digest = exp.shared.bench.digest
    source_hash = digest(args.source / "adapter_model.safetensors")
    if source_hash != previous["source_hash"]:
        raise ValueError("Use the original DART adapter, not the trained control")
    if digest(args.source / "adapter_config.json") != previous["source_config_hash"]:
        raise ValueError("Source adapter configuration differs")
    if digest(args.cache / "model.safetensors") != exp.shared.bench.MODEL_HASH:
        raise ValueError("Official weights differ")

    data_bytes = (args.baseline / "data.json").read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != previous["data_hash"]:
        raise ValueError("Baseline data changed")
    data = json.loads(data_bytes)

    identity = dict(
        previous,
        format="mamba-inout-sft-v1",
        driver_hash=digest(Path(__file__)),
        helper_hash=digest(Path(exp.__file__)),
        targets=["in_proj", "out_proj"],
        contrastive_weight=0.,
        training="Same examples and optimizer updates; compute is not equal",
    )
    args.run.mkdir(parents=True, exist_ok=True)
    manifest = args.run / "identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Existing run settings differ; use a new directory")
    elif any(args.run.iterdir()):
        raise FileExistsError("Use an empty output directory")
    exp.shared.write_json(manifest, identity)

    tokenizer = exp.AutoTokenizer.from_pretrained(
        str(args.cache), local_files_only=True
    )
    config = exp.MambaConfig.from_pretrained(
        str(args.cache), local_files_only=True
    )
    config.use_mambapy, config.use_cache = True, False
    backbone = exp.MambaForCausalLM.from_pretrained(
        str(args.cache),
        config=config,
        local_files_only=True,
        torch_dtype=exp.torch.float32,
        use_safetensors=True,
    )
    lora = LoraConfig.from_pretrained(
        str(args.source), local_files_only=True
    )
    if set(lora.target_modules) != {"out_proj"} or lora.r != 8:
        raise ValueError("Expected rank-8 out_proj source")
    lora.target_modules = {"in_proj", "out_proj"}
    lora.inference_mode = False

    exp.torch.manual_seed(42)
    model = get_peft_model(backbone, lora)
    old = load_peft_weights(str(args.source), device="cpu")
    exp.set_peft_model_state_dict(model, old)
    initial = {
        k: v.detach().clone()
        for k, v in exp.get_peft_model_state_dict(model).items()
    }
    if any(
        k not in initial or not exp.torch.equal(v, initial[k])
        for k, v in old.items()
    ):
        raise ValueError("Original adapter weights were not copied exactly")
    added = {k: v for k, v in initial.items() if k not in old}
    if not added or any("in_proj" not in k for k in added):
        raise ValueError("Unexpected new modules")
    if any(
        exp.torch.count_nonzero(v).item()
        for k, v in added.items() if "lora_B" in k
    ):
        raise ValueError("New LoRA projections must initially have zero effect")

    params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in params), flush=True)
    print("Only the new in/out arm runs; original control reused.", flush=True)

    stopped = [False]
    handler = signal.signal(
        signal.SIGINT, lambda *_: stopped.__setitem__(0, True)
    )
    try:
        step, _ = exp.arm(
            model, initial, params, data, args,
            "inout", 0., identity, stopped
        )
        if step == args.updates and not stopped[0]:
            selected = exp.torch.load(
                args.run / "inout/best.pt",
                map_location="cpu",
                weights_only=True,
            )
            exp.set_peft_model_state_dict(model, selected["adapter"])
            result = dict(
                test=exp.evaluate(model, data["test"]),
                generations=exp.generations(model, tokenizer, data["test"]),
            )
            results = dict(
                out_proj=report["results"]["control"],
                inout=result,
            )
            exp.shared.write_json(
                args.run / "comparison.json",
                dict(identity=identity, results=results),
            )
            for name, item in results.items():
                print("\n", name, "TEST:", item["test"], flush=True)
                for row in item["generations"]:
                    print(
                        row["family"], "|", row["reference"],
                        "->", row["generated"], flush=True
                    )
        if digest(args.source / "adapter_model.safetensors") != source_hash:
            raise ValueError("Source adapter changed")
        print("Saved:", args.run, flush=True)
    finally:
        signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":
    main()
