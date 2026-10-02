# Frozen PiPER inference on RTX 5090

Run the verified PiPER checkpoint in a separate Python process using PyTorch
2.7.0 + CUDA 12.8 and torchvision 0.22.0. Reuse the installed RTX 5090 runtime
through an isolated virtual environment; keep the Isaac Python environment
unchanged. The model uses PyTorch SDPA, so FlashAttention, DeepSpeed, LeRobot and
the full training dependency set are unnecessary.

The reference H200 environment uses PyTorch 2.6.0, torchvision 0.21.0 and
Transformers 4.57.0. Keep its model and processor dependency versions while
using the local CUDA 12.8 PyTorch build. Existing Transformers 5.x installations
are not interchangeable with this checkpoint loader; the virtual environment
below installs the matching 4.57.0 version locally. Different GPU architectures
and PyTorch kernels can introduce numerical differences; run a frozen-policy
comparison before attributing task-success changes to RL.

## Isolated environment

Set these paths for the local checkout and existing PressB runtime:

```bash
PRESSB_ROOT=/home/pengguanqi/Workspace/Research/PressB
VLA_REPO="$PRESSB_ROOT/src/VLA-JEPA"
INFER_ENV="$PRESSB_ROOT/.conda/envs/vlajepa-inference"

"$PRESSB_ROOT/.conda/envs/pressb/bin/python" -m venv \
  --system-site-packages "$INFER_ENV"

cat > "$INFER_ENV/torch-constraints.txt" <<'EOF'
torch==2.7.0+cu128
torchvision==0.22.0+cu128
EOF

"$INFER_ENV/bin/python" -m pip install \
  -c "$INFER_ENV/torch-constraints.txt" \
  -r "$VLA_REPO/requirements-piper-inference-cu128.txt"
```

Use the virtual environment's absolute Python path for every subsequent command.
Pip adds matching model libraries inside this environment; the inherited
torch/torchvision packages remain in the parent runtime. Do not install
`requirements-piper.txt` here: that training requirements file pins the older
PyTorch 2.6 runtime used on H200.

A CPU-only import check does not load checkpoint weights or claim successful
GPU inference:

```bash
CUDA_VISIBLE_DEVICES='' "$INFER_ENV/bin/python" - <<'PY'
import torch, torchvision, transformers, diffusers, timm, accelerate
from transformers import Qwen3VLForConditionalGeneration, VJEPA2Model
assert torch.__version__ == '2.7.0+cu128'
assert torchvision.__version__ == '0.22.0+cu128'
assert transformers.__version__ == '4.57.0'
print(torch.__file__, torchvision.__file__)
print('CPU imports passed')
PY
```

## Minimal offline assets

Keep the original run/checkpoint hierarchy and copy these files without editing
their contents:

```text
run/
  config.yaml
  action_schema.json
  split.json
  training_contract.json
  processor/                         # every saved processor file
  checkpoints/step_010600/
    manifest.json
    model.pt
```

The verified step-10600 `model.pt` contains all four model components and is
10,268,119,458 bytes. Its expected SHA256 is
`2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312`.
`training.pt` is unused by inference; skipping it saves 16,266,376,994 bytes.
No dataset or additional checkpoint step is required.

Both backbone constructors use `init_from_config=true`, so their base weights
are unnecessary. The original constructors still require the following small
configuration and processor assets:

```text
Qwen3-VL-2B-Instruct/
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  preprocessor_config.json
  video_preprocessor_config.json
  chat_template.json

vjepa2-vitl-fpc64-256/
  config.json
  video_preprocessor_config.json
```

Use the actual chat-template filename from the original backbone directory;
copy its matching `chat_template.*` file. The Qwen model factory selects its
implementation by the `Qwen3-VL` substring in the local path, so retain that
directory name. The VJ encoder also constructs an `AutoVideoProcessor`; its
video processor configuration cannot be omitted even though online inference
uses the Qwen encoder and flow head.

Recommended local placement is
`$PRESSB_ROOT/.cache/vlajepa/{Qwen3-VL-2B-Instruct,vjepa2-vitl-fpc64-256}` and
`$PRESSB_ROOT/.cache/vlajepa/piper_panel_stratified_press30hz_pose9_3epochs_20260929`.
Pass relocated backbone paths through `PiperPolicy(base_vlm=...,base_encoder=...)`;
do not rewrite the saved run `config.yaml` or training provenance.

## Frozen model loading and deployment checks

After the transfer is complete and the checkpoint SHA has been verified, load
the model on the intended GPU. `CUDA_VISIBLE_DEVICES=1` maps physical GPU 1 to
process-local `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$INFER_ENV/bin/python" - <<'PY'
from pathlib import Path
import sys
root = Path('/home/pengguanqi/Workspace/Research/PressB')
sys.path.insert(0, str(root / 'src/VLA-JEPA'))
from starVLA.inference.piper_policy import PiperPolicy
assets = root / '.cache/vlajepa'
policy = PiperPolicy(
    assets / 'piper_panel_stratified_press30hz_pose9_3epochs_20260929/checkpoints/step_010600',
    device='cuda:0',
    base_vlm=assets / 'Qwen3-VL-2B-Instruct',
    base_encoder=assets / 'vjepa2-vitl-fpc64-256',
)
assert all(not p.requires_grad for p in policy.model.parameters())
print(policy.provenance)
PY
```

Strict loading retains checkpoint checksum, saved vocabulary, state/action
schema and training-contract checks. The complete JEPA components are loaded
even though they do not execute during online policy inference.

For an HTTP inference wrapper, expose `base_vlm` and `base_encoder` overrides to
the same constructor. Select a separate test port (for example 19891) while the
H200 fallback remains available. Keep RGB PNG transport and the same model
resolution, prompt, camera ordering and seeded noise convention during the
deployment comparison.

Verify recorded RGB/state inputs against the H200 service, both residual base
actions and explicit-noise flow decoding. Measure native model throughput and
the full simulator/learner pipeline separately. Sharing physical GPU 1 with
Isaac may fit in memory but also shares GPU execution time; successful loading
alone does not establish faster training. Record peak memory under simultaneous
simulation, rendering and the intended inference batch size.
