# Cosmos-Dreams offline parity runner

This runner accepts the reference interactive jsonl plus its optional pickle
payload. A record may contain `prompt`/`ai_caption`, `input_video`/`video`/`image`,
`action`, `fps`/`conditioning_fps`, and `domain_id`, or point to a pickle with `pickle_path`, `pkl_path`, or
`data_path`. The first source frame is used as the causal prefix; action rows
are normalized from the exported deployment stats and padded to 64 dimensions.

```bash
python examples/offline_inference/cosmos_dreams/cosmos_dreams.py \
  --model /checkpoints/cosmos-dreams-diffusers \
  --jsonl /data/reference_samples.jsonl \
  --sample-index 0 \
  --num-frames 601 \
  --seed 42 \
  --output cosmos_dreams_sample_0.mp4
```

Use `--output-type latent --output sample_0.pt` for the pre-VAE parity gate.
Full rollouts send both `reset=True` and `close_session=True`, preventing the
default session from leaking history into the next sample.
