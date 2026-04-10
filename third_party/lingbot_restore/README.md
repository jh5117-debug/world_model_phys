These files are a minimal recovery snapshot for the external LingBot runtime code
used on H20 under `/home/nvme03/workspace/world_model_phys/code`.

They are not a full copy of the collaborator repo. The goal is only to restore the
small set of runtime files that were accidentally truncated to zero bytes during
debugging:

- `code/finetune_v3/lingbot-csgo-finetune/eval_batch.py`
- `code/finetune_v3/lingbot-csgo-finetune/inference_csgo.py`
- `code/lingbot-world/wan/image2video.py`
- `code/lingbot-world/wan/modules/attention.py`

Use `scripts/restore_external_lingbot_code.sh` from this repo to copy these files
back into the external H20 code tree before running LingBot base or Stage-1
inference.
