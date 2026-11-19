import json
from pathlib import Path

import compare_mamba_faithfulness as exp
from peft import PeftModel

torch = exp.torch
torch.set_num_threads(8)
cache = Path("runs/_models/mamba130m_1e76775f")
baseline = Path("runs/mamba130m_faithfulness_training")
output = Path("runs/mamba130m_all_pairs_review.json")

adapters = {
    "out_proj": baseline / "control/best_adapter",
    "inout": Path("runs/mamba130m_inout_sft/inout/best_adapter"),
}
digest = exp.shared.bench.digest
if digest(cache / "model.safetensors") != exp.shared.bench.MODEL_HASH:
    raise ValueError("Official weights changed")

data_bytes = (baseline / "data.json").read_bytes()
data = json.loads(data_bytes)
identity = {
    "data_sha256": digest(baseline / "data.json"),
    "adapters": {
        name: {
            "weights": digest(path / "adapter_model.safetensors"),
            "config": digest(path / "adapter_config.json"),
        }
        for name, path in adapters.items()
    },
    "generation": {"max_new_tokens": 64, "do_sample": False},
}
state = {"identity": identity, "rows": []}
if output.exists():
    state = json.loads(output.read_text(encoding="utf-8"))
    if state["identity"] != identity:
        raise ValueError("Existing results use different inputs or adapters")

done = {(r["mode"], r["unit"], r["variant"]) for r in state["rows"]}
tokenizer = exp.AutoTokenizer.from_pretrained(
    str(cache), local_files_only=True
)

for mode, adapter in adapters.items():
    expected = len(data["test"]) * 2
    completed = sum(r["mode"] == mode for r in state["rows"])
    if completed == expected:
        print(mode, "already complete:", completed, flush=True)
        continue

    config = exp.MambaConfig.from_pretrained(
        str(cache), local_files_only=True
    )
    config.use_mambapy, config.use_cache = True, False
    backbone = exp.MambaForCausalLM.from_pretrained(
        str(cache),
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(
        backbone, str(adapter), is_trainable=False
    )
    model.eval()

    with torch.inference_mode():
        for index, unit in enumerate(data["test"]):
            for variant, row in enumerate(unit["cases"]):
                key = (mode, index, variant)
                if key in done:
                    continue
                x = tokenizer(
                    row["prompt"], return_tensors="pt"
                )["input_ids"]
                generated = model.generate(
                    input_ids=x,
                    attention_mask=torch.ones_like(x),
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(
                    generated[0, x.shape[1]:],
                    skip_special_tokens=True,
                )
                state["rows"].append({
                    "mode": mode,
                    "unit": index,
                    "variant": variant,
                    "family": unit["family"],
                    "facts": row["prompt"],
                    "reference": row["positive"],
                    "generated": text,
                })
                exp.shared.write_json(output, state)
                done.add(key)
                completed += 1
                print(
                    mode, str(completed) + "/" + str(expected),
                    unit["family"], "|", text, flush=True
                )

    del model, backbone

print("\nSaved:", output)
print("Review fidelity, completeness and unsupported additions.")
print("No automatic semantic accuracy is claimed.")
