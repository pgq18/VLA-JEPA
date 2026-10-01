# PiPER experiment status

The PiPER path reads press-only LeRobot v3 data with global and wrist RGB cameras.
It learns absolute TCP XYZ in `base_link` (metres) plus the first two rows of a
rotation matrix as 6D rotation. The constant gripper opening is not learned.
See [the adapter and training guide](piper_lerobot.md) for installation, action
alignment, checkpoint validation and inference.

## Completed panel-offset experiment

Run: `piper_panel_stratified_press30hz_pose9_3epochs_20260929`.
Data: `piper_elevator_lerobot_panel_stratified_press_30hz`, 1,200 episodes,
12 floor tasks, 30 Hz, two 640×480 cameras and 306,830 frames. Each episode ends
at the first sampled target-button light-on frame. Panel offsets cover X ±10 mm
and Y ±25 mm using a stratified grid; X is the front/back direction and Y is the
left/right direction in this scene.

Training started from the official pretrained checkpoint, with 1,080 episodes /
276,063 frames for training and 120 episodes / 30,767 frames for validation.
Four H200 GPUs with a global batch of 64 completed 4,314 updates per epoch,
12,942 total updates and three epochs. The supervisor exited with code 0 at
2026-09-30 10:14:03 Asia/Shanghai, after 5 h 15 min 2 s.

| Checkpoint | Step | Held-out first-action XYZ error | Rotation error |
| --- | ---: | ---: | ---: |
| Best | 10,600 | 6.135 mm | 0.330° |
| Last | 12,942 | 7.779 mm | 0.305° |

These are offline errors on 600 fixed held-out anchors, not task success rates.
Best selection uses position error in metres plus 0.1 times rotation error in
radians. Both checkpoint model and optimizer files were independently checked
against their saved SHA-256 manifests.

Best model SHA-256:
`2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312`.
Dataset export-manifest SHA-256:
`7e7d30160b084b96074c2e0a198000607f96d5704dc57f9a2afa933da899c102`.
Split SHA-256:
`7aeed6a5b0ef1ff4d04bcd9e8e736329cf4a7180660350264dcca3e5e6c82210`.

On the experiment host, complete run artifacts remain outside Git under
`/data/scratch/pengguanqi/VLA-JEPA-runs/` followed by the run name above.
Model assets, datasets, checkpoints, setup reports, logs and videos are not
published with the source repository.

## Isaac Sim closed-loop evaluation

The best checkpoint was evaluated in the PressB simulator with 12 floors × five
panel positions (centre and four X/Y corners), one attempt per condition. Each
attempt used live cameras, measured TCP state and the floor task text. Policy
inference ran at 30 Hz with seven-action chunks; physics ran at 120 Hz. The
controller used joint interpolation and a three-point causal moving average.
The simulated timeout was 15 seconds per attempt, and physics paused while
waiting for remote inference. No expert trajectories or button coordinates were
used to correct model actions.

| Position | Successful target presses / attempts |
| --- | ---: |
| Centre | 1 / 12 |
| X−, Y− | 1 / 12 |
| X−, Y+ | 2 / 12 |
| X+, Y− | 1 / 12 |
| X+, Y+ | 0 / 12 |
| Total | 5 / 60 (8.3%) |

There were 27 wrong-button presses and 28 timeouts, with no unexpected
collisions. All wrong-button presses were on the next floor above the requested
floor. The independent recording audits passed, covering physical press
criteria, state/action records, image hashes, frame counts and video timestamps.
They do not certify image semantics or that the illuminated edge is visible from
every camera. Each condition has only one trial; these results do not establish
a statistically reliable general success rate.

The source checkout of the companion PressB project retains the full report at
`outputs/policy_eval_step10600_stratified/center_corners_v1/summary.md` and its
per-episode recordings. Those generated artifacts are separate from source
control. The current model's closed-loop performance remains insufficient for
reliable button pressing.

## Source checks before publication

The dedicated environment is described by `requirements-piper.txt`. Run its
CPU tests and real-dataset integration checks without allocating GPU memory:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PIPER_TEST_ROOT=/path/to/piper_elevator_lerobot_panel_stratified_press_30hz \
python -m pytest tests/test_piper_lerobot.py tests/test_piper_model_masks.py \
  tests/test_piper_trainer.py tests/test_piper_policy.py \
  tests/test_eval_piper_checkpoint.py tests/test_serve_piper_policy.py -q
```

Before publication, this command passed on H200: **89 tests and 16 subtests**,
including integration checks against the real stratified dataset, with CUDA
disabled for the test process. The source-only credential/size scan and
`git diff --check` also passed.

Historical results are kept in [the initial fine-tuning report](piper_finetune_result.md)
and [the fixed-panel continuation notes](piper_continuation.md).
