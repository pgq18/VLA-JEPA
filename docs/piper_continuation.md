# 从第 2000 步继续到累计 3 个 epoch

本次续训使用 `piper_press30hz_pose9_20260929/checkpoints/step_002000`，在新目录 `runs/piper_press30hz_pose9_3epochs_from2000_20260929` 保存结果。原始 2000 步检查点和评估证据保留。

训练仍使用 1080 条 episode、275687 个起始帧，120 条 episode 仅用于验证。4 张 H200，每卡 batch=16，梯度累积=1；每个 epoch 有 4308 个优化器更新，包含末尾每卡 10 个样本的小批次。累计 3 个 epoch 对应第 **12924 步**，从第 2000 步还需 **10924 步**。分布式采样器每个 epoch 补齐一个重复样本，各原始训练样本均被遍历。

`--continue-from` 用于创建明确记录来源的新训练阶段。它恢复模型权重、AdamW 状态、各 rank 随机数状态以及 epoch 内采样位置。恢复时从采样索引直接跳过已训练数据，不重新解码前 2000 批视频。四元数转矩阵前两行的 6D 表示、基座坐标系米制 XYZ 和夹爪不参与学习的约定保持一致。

原始学习率在第 2000 步已降到基准值的 10%。新阶段先用 100 步从该值线性升至原基准值（Qwen=1e-5、JEPA predictor=5e-4、action head=1e-4），随后余弦下降到各自基准值的 10%。优化器动量保留；这是一段明确记录的新学习率计划。新目录的 `best` 只比较本阶段检查点，父检查点的成绩记录在续训来源文件中，可另外比较。

在 H200 的仓库根目录，实际训练参数为：

```bash
bash scripts/run_piper_ft.sh \
  --run-id piper_press30hz_pose9_3epochs_from2000_20260929 \
  --continue-from runs/piper_press30hz_pose9_20260929/checkpoints/step_002000 \
  --target-epochs 3
```

后台启动使用现有 `piper_setup/supervise.py`，输入输出重定向并创建独立进程会话。监督进程写入退出状态，无需保持 SSH shell 或人工持续查看。此命令在已启动后不要重复执行；查看进展：

```bash
tail -n 3 runs/piper_press30hz_pose9_3epochs_from2000_20260929/train.jsonl
cat piper_setup/logs/piper_press30hz_pose9_3epochs_from2000_20260929.status.json
```

如果将来需要从本阶段的检查点恢复，使用保存的配置和严格 `--resume`，不再次建立学习率热身阶段：

```bash
bash scripts/run_piper_ft.sh \
  --config runs/piper_press30hz_pose9_3epochs_from2000_20260929/config.yaml \
  --resume runs/piper_press30hz_pose9_3epochs_from2000_20260929/checkpoints/last
```

`runs` 指向 `/data/scratch/pengguanqi/VLA-JEPA-runs`。每次保存后保留本阶段的 `best` 和 `last`，并以临时目录完成写入和 SHA 校验后发布检查点；预留的磁盘空间覆盖 `best`、`last` 与一次正在写入的检查点。训练结束写入 `completed.json`，应显示 `step=12924, epoch=3, batch_in_epoch=0`。该结束状态由后台任务产生，不代表启动检查时已经完成。
