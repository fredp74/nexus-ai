# Nexus AI

Experimental grounded data-to-text generation using the official Mamba-130M model and LoRA adapters on CPU. The pretrained base model is not trained by this project.

## Current workflow

Run from the repository root. `python train_mamba_reviewed_dart.py` resumes the latest experiment. Local weights, adapters and datasets are required and excluded from Git.

Official model revision: `1e76775f628fbf1350fbe4dbb3d971ba64af25a1`. Adapters target `in_proj` and `out_proj`. The reviewed DART pilot uses 48 training, 16 validation and 16 test graphs, with synthetic replay.

Evaluation includes NLL and inspection of generated text. Relation inversions, omissions and unsupported additions still occur.

Install dependencies with `python -m pip install -r requirements.txt`.

## Credits

- https://github.com/state-spaces/mamba
- https://huggingface.co/state-spaces/mamba-130m-hf
- https://github.com/alxndrTL/mamba.py
- https://github.com/furiosa-ai/ssm-peft
- https://github.com/Yale-LILY/dart

Upstream code, weights and datasets retain their respective licenses.
