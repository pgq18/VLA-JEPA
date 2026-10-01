# PiPER LeRobot fine-tuning and inference

This repository includes the native LeRobot v3 adapter, distributed fine-tuning,
strict checkpoint inference and a localhost policy service for the PiPER elevator
project. The latest panel-offset run completed three epochs (12,942 steps); its
best checkpoint is step 10,600. In the subsequent Isaac Sim evaluation it pressed
the requested button in 5 of 60 trials. See [current experiments and evaluation](piper_status.md)
for the data split, checkpoint identity and limitations. The original 2,000-step
run is preserved as [historical measured results](piper_finetune_result.md).

The native adapter reads the immutable `piper_elevator_lerobot_press_30hz`
LeRobot v3 dataset: 1,200 episodes, 12 floor tasks (24–35), 100 episodes per
floor, and 306,230 frames at 30 Hz. Each episode ends at the **first sampled
target-button light-on frame**, including that frame. The stored terminal
action repeats the previous planned target, which has already been executed.
There is no post-press retreat or return-home supervision in this dataset.

## Environment and files

Use the dedicated `requirements-piper.txt`, not the full generic training
requirements: this path does not need FlashAttention, DeepSpeed, or LeRobot.
On the prepared H200 host the Python 3.10.20 environment is
`/home/pengguanqi/miniconda3/envs/vlajepa-piper`:

```bash
source /home/pengguanqi/miniconda3/etc/profile.d/conda.sh
conda activate vlajepa-piper
```

On H200, `runs` is a symlink to
`/data/scratch/pengguanqi/VLA-JEPA-runs`. Keep checkpoints there: this account
has a 100 GiB hard quota on `/home`, even when `df` reports free disk space.
Check `quota -s` as well as filesystem space. A complete checkpoint currently
needs about 25 GiB for model and optimizer together; best, last, and an in-flight
save can require three copies. Model files, wheel downloads, and setup logs are
excluded from Git.

For a fresh compatible CUDA host, install PyTorch 2.6.0 and torchvision 0.21.0
from the CUDA 12.4 wheel index first, then the dedicated requirements:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-piper.txt
```

That environment pins Transformers 4.57.0, NumPy 1.26.4, PyArrow 18.1.0,
and PyAV 15.1.0. The native adapter itself needs only NumPy, Pillow, PyArrow,
and PyAV. It does not import LeRobot or require conversion to another dataset
format. Model assets and the official pretrained checkpoint must be fully
downloaded before training can start.

Keep the published dataset read-only. Set these paths in
`scripts/configs/vlajepa_piper_ft.yaml`, or supply the corresponding `--set`
overrides:

* `datasets.vla_data.data_root_dir`: published dataset root.
* `framework.qwenvl.base_vlm`: local Qwen3-VL model config/processor directory.
* `framework.vj2_model.base_encoder`: local V-JEPA2 config/processor directory.
* `trainer.pretrained_checkpoint`: VLA-JEPA pretrained checkpoint file.
* `run_root_dir`: a writable directory outside the dataset.

Both backbone `init_from_config` values are `true`: construct the matching
architecture, then load the VLA-JEPA checkpoint. They do **not** authorize
inference with random weights. Fine-tuning explicitly permits only the named
input/output layer changes needed for the 9D interfaces; its loading report
records those changes. Inference requires every saved trained-model tensor.

## Poses and temporal alignment

Original state/action rows have eight values:
`[x_m, y_m, z_m, qw, qx, qy, qz, gripper_width_m]`. State is the measured TCP
pose, and a nonterminal action is the absolute target for the next 30 Hz
sample. Both are expressed in **PiPER `base_link`**, at `gripper_tcp`
(`link6` local +Z offset 0.1358 m), not the pressing tool tip.

The model uses nine values:
`[x_m, y_m, z_m, r00, r01, r02, r10, r11, r12]`.
The rotation is PyTorch3D's `R[:2, :].reshape(6)` convention: the first two
**rows**, flattened by row. XYZ retains the original float32 values in metres;
there is no normalization, relative-action conversion, coordinate transform,
or IK. Gripper width is excluded from **both** the model state and action.

At anchor frame `i`:

| Field | Contents |
| --- | --- |
| `image` | Current RGB images, ordered **global, wrist** |
| `state` | Current pose, shape `[1,9]` |
| `action` | Stored actions `i..i+6`, shape `[7,9]` |
| `video` | Frames `i..i+7`, including current, shape `[2,8,224,224,3]` |
| `action_is_pad` / `video_is_pad` | True only for requested rows beyond the episode end |

Indices beyond the end repeat the final row, stay within that episode, and
are masked out of losses. The actual terminal row is valid supervision. At
the last anchor, action padding is `[false,true,true,true,true,true,true]`.
The world-model loss excludes a future temporal tubelet if either of its two
frames is padding. `sample_stride` only selects training anchors; it does not
change the 30 Hz target spacing. Every episode's final anchor remains included.

Camera videos are concatenated v3 files. The reader uses each episode's
per-camera `from_timestamp` plus frame timestamps; it does not assume the
first frame of each episode is at video time zero. Decoded PTS must be within
half a frame of the requested timestamp. Original images are 640×480;
training uses 224×224 RGB with bilinear resizing.

## Train and resume

From the repository root, in the CUDA environment:

```bash
PIPER_GPUS=4 bash scripts/run_piper_ft.sh --run-id piper_pose9_run01
```

The adapter writes a reproducible, whole-episode split at `run/split.json`.
Seed 42 gives **1,080 train / 120 validation episodes**, exactly 90/10 for
each floor, with no episode shared between splits. It binds the split to the
export-manifest SHA and records camera order. The published source has
275,687 train and 30,543 validation frames under this split.

The trainer saves configuration, processor, action schema, dataset/split
hashes, training contract, pretrained loading report, validation anchors,
and physical validation metrics. `checkpoints/best` and `checkpoints/last`
are aliases to complete step directories. `model.pt` contains the full plain
state dict; `training.pt` holds optimizer/scheduler/RNG/progress for resume.
Validation calls `predict_action`, reports position error in metres and
orientation geodesic error in degrees, and excludes padding. Best checkpoint
selection uses first-action position error plus 0.1 times first-action
orientation error in radians. A low offline error is not a simulation success
rate.

Resume with the same world size and training contract:

```bash
PIPER_GPUS=4 bash scripts/run_piper_ft.sh --run-id piper_pose9_run01 \
  --resume /path/to/runs/piper_pose9_run01/checkpoints/last
```

Use a new run ID for changes to sampling, seed, optimizer, schedule, or
validation selection. The run directories and split manifests belong outside
the immutable dataset.

## Infer from a trained checkpoint

```python
from PIL import Image
from starVLA.inference.piper_policy import PiperPolicy

policy = PiperPolicy(
    "/path/to/runs/piper_pose9_run01/checkpoints/best",
    device="cuda:0",
    controller_gripper_width_m=0.008,
)
result = policy.predict(
    global_image=Image.open("global.png"),
    wrist_image=Image.open("wrist.png"),
    state=measured_state8,  # one raw 8D vector, or one measured 9D vector
    task="Press 35 floor.",
    seed=42,
)
raw_pose9 = result["raw_pose9"]  # [7,9], native model output, unnormalized
pose8 = result["pose8"]          # [7,8], xyz + unit wxyz + fixed gripper
```

The loader verifies checkpoint completeness and model SHA, binds the schema
to the dataset/split/training contract, checks saved config semantics, checks
the saved processor's vocabulary/token IDs, and loads all four model components
with `strict=True`, preserving the saved tensor dtypes. Shared parameter aliases
must also contain identical saved values. It disables unused Qwen decoder
KV caches to match training validation, calls `eval()`, and disables gradients. Optimizer state
is not needed for inference. A moved run can use `base_vlm` / `base_encoder`
path overrides, but needs the same matching local configurations and processor
assets. A Qwen3-VL directory name must retain `Qwen3-VL`, as the repository
selects its VLM backend from that path.

The old upstream return key `normalized_actions` is read internally; this
PiPER schema explicitly says `normalization="none"`. The returned raw 9D
array is not normalized. Conversion of its rotation uses Gram–Schmidt to
produce SO(3), then unit **wxyz**; a zero or parallel pair of predicted rows
raises `ValueError`. XYZ is untouched. The default appended **0.008 m total
gripper opening is an explicit controller constant**, matching this collection,
and is not learned or inferred from the predicted action. It can be overridden
explicitly via `controller_gripper_width_m`. The entry point does not execute
IK, joint commands, simulation, or real hardware.

To compare an actual sample to its recorded action chunk without simulation:

```bash
python -m starVLA.inference.piper_policy \
  --checkpoint /path/to/runs/piper_pose9_run01/checkpoints/best \
  --dataset-root /path/to/piper_elevator_lerobot_press_30hz \
  --source-episode 1199 --frame 280 --device cuda:0 \
  --output /path/to/reports/episode1199_frame280.json
```

Use `--base-vlm` / `--base-encoder` when the saved backbone paths are absent on
the inference host. The report records predictions, targets, padding,
unpadded errors, model/metadata/code hashes, fixed opening, and sample identity.
An existing output is not overwritten. Source episode access works for
either saved split; validation comparisons should use its held-out episodes.

For an independent check across all 12 floors, load the model **once** and
evaluate 36 held-out anchors:

```bash
python -m scripts.eval_piper_checkpoint \
  --checkpoint /path/to/runs/piper_pose9_run01/checkpoints/best \
  --dataset-root /path/to/piper_elevator_lerobot_press_30hz \
  --device cuda:0 --output /path/to/reports/piper_pose9_run01_heldout36.json
```

The script reads the run's saved split, takes the lowest source episode ID
among the ten held-out episodes of each floor, and checks frames `0`,
`round((N-1)/2)`, and `N-1`. It uses the trainer's per-anchor evaluation seed
formula by default. This selection is fixed before inspecting predictions.
It reports unpadded Euclidean XYZ errors in metres, SO(3) geodesic errors in
degrees, first-action errors, and a current-state-copy baseline. Each anchor
includes its raw prediction, restored quaternion, target, padding, and fixed
gripper check. A failure remains in the JSON and causes a nonzero process exit.

The report also records checkpoint/data/code hashes, loaded model dtype counts,
`features_only`/`eval` status, and GPU peak memory. Its `success` field means
all 36 selected predictions are finite and follow the schema; it applies no
accuracy threshold and does not measure button-press or robot success.

## Verification performed

```bash
PIPER_TEST_ROOT=/home/pengguanqi/Datasets/piper_elevator_lerobot_press_30hz \
  python -m pytest tests/test_piper_lerobot.py tests/test_piper_model_masks.py \
  tests/test_piper_trainer.py tests/test_piper_policy.py \
  tests/test_eval_piper_checkpoint.py -q
```

All **77 tests passed on H200** in the pinned Python 3.10/CUDA environment,
including the real-dataset integration checks. Real four-GPU tests also
completed 2 optimizer steps at batch 4/GPU and 20 steps at batch 16/GPU,
saved full checkpoints, and confirmed updates to the language model, JEPA
predictor, and action head. The 20-step checkpoint was independently restored
on a GPU and produced finite, schema-correct predictions on all 36 held-out
anchors. These short runs verify execution and checkpoint reuse; their pose
accuracy is not a task-success claim. The full fine-tuning results are recorded
in the formal run's validation and acceptance reports.

The 12 native-adapter CPU tests passed against the real published dataset.
They cover rotation conventions/round-trips, deterministic stratified splits,
nonzero v3 video offsets against independent decoding, end clamping/masks,
arbitrary-frame access, caching/pickling, and a two-worker DataLoader. In the
local validation environment these used NumPy 2.2.6, PyArrow 21.0.0, PyAV
15.1.0, Pillow 12.3.0, and CPU PyTorch 2.10.0; matching these exact versions
is not required by the adapter. A single local CPU process measured about
160 sequential or 25 random samples/s at 224 px (128 samples each; this is
data reading, not model throughput).

The 28 inference CPU tests passed: explicit gripper restoration, 9D/8D pose
conversion, degenerate rotation rejection, exact schema/contract checks,
checkpoint/hash failures, strict tensor loading, tied aliases, processor
checks, mocked image/state inference, and padding-aware comparisons. These
tests do **not** claim a GPU model forward, completed fine-tuning, or successful
robot control. Actual training and checkpoint inference results belong in the
run's logs and generated prediction reports.

The independent 36-anchor evaluator has 11 passing CPU tests for held-out
selection, leakage/missing-floor rejection, terminal padding, physical error
calculations, output corruption, repeated calls on one model object, and
failure reporting. Those use mocked predictions; an actual GPU evaluation
report can only be produced after a trained checkpoint exists.
