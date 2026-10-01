# PiPER VLA-JEPA 微调报告：piper_press30hz_pose9_20260929

训练完成：2000 个优化器更新步骤，4 卡 DDP；独立重载验收通过。训练与推理进程退出码均为 0，验收确认运行代码未改变。

## 模型、数据与训练设置

使用官方 VLA-JEPA 预训练权重，严格加载 1613 个张量；仅 4 个输入/输出接口按 PiPER 9 维表示重新初始化。实际训练参数 2,033,214,217 个，语言骨干参数变化已有记录。

数据按完整 episode 分为 1080 条训练、120 条验证；24–35 层共 12 个任务，每层 90/10 条。验证与训练 episode 不重叠。每条数据在首次按键亮灯采样帧结束。

state/action 均为机械臂 base_link 坐标系中的 gripper_tcp 绝对位姿：XYZ 保持米制原值；旋转使用 R 前两行按行展开的 6D 表示。共 9 维，不做坐标归一化，夹爪开度不参与学习。

采样率 30 Hz，动作窗口 7 帧、JEPA 视频窗口 8 帧。双相机顺序为 global、wrist，训练输入为 224×224 RGB。

每卡 batch=16，GPU=4，梯度累积=1，有效全局 batch=64；总计 2000 步，warmup=100，每 200 步评估。记录的 GPU 型号为 NVIDIA H200 NVL。

远端实际测试日志：**77 passed**，耗时 8.88 秒。环境记录：PyTorch 2.6.0，Transformers 4.57.0。

## 最佳检查点的离线验证

最佳步骤为 **2000**。验证包含 120 条保留 episode 的 600 个固定 anchor，每层 50 个。以下均为生成动作窗口中第一个动作相对真实目标的平均误差；旋转先投影到 SO(3)。

| 方法 | 位置误差（mm） | 旋转误差（°） |
| --- | ---: | ---: |
| 最佳模型，第 2000 步 | 11.738 | 0.641 |
| 保持当前 state 的基线 | 3.404 | 0.082 |

模型的 flow loss=0.001637；JEPA 辅助损失（已乘 0.1）=0.108276。

| 楼层 | anchor 数 | 模型位置（mm） | 模型旋转（°） | 基线位置（mm） | 基线旋转（°） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 | 50 | 12.944 | 0.651 | 3.179 | 0.075 |
| 25 | 50 | 13.000 | 0.633 | 3.233 | 0.076 |
| 26 | 50 | 12.140 | 0.617 | 3.333 | 0.078 |
| 27 | 50 | 10.598 | 0.596 | 3.550 | 0.080 |
| 28 | 50 | 10.767 | 0.547 | 3.672 | 0.082 |
| 29 | 50 | 13.418 | 0.575 | 3.515 | 0.091 |
| 30 | 50 | 9.952 | 0.688 | 3.117 | 0.079 |
| 31 | 50 | 9.810 | 0.703 | 3.257 | 0.081 |
| 32 | 50 | 10.861 | 0.621 | 3.313 | 0.082 |
| 33 | 50 | 11.177 | 0.653 | 3.586 | 0.085 |
| 34 | 50 | 11.759 | 0.758 | 3.657 | 0.086 |
| 35 | 50 | 14.433 | 0.648 | 3.432 | 0.092 |

在首动作位置误差上，模型高于保持当前 state 的基线。30 Hz 相邻目标的变化较小，基线结果应与模型结果一起解读。

## 独立重载验证与适用范围

从磁盘严格重载最佳检查点一次，覆盖 12 个楼层、每层一条验证 episode 的首/中/末帧，共 **36/36 个 anchor 通过接口检查**。首动作位置误差为 13.750 mm，旋转误差为 0.558°；同组 state-copy 基线为 3.393 mm、0.071°。

此项通过表示输出有限、位姿与四元数转换正确、夹爪固定且检查点来源一致，未设置机器人任务成功率门槛。**本次未执行训练后策略的 Isaac Sim 闭环按键成功率测试，不能把离线误差或 36/36 接口通过率当作按键成功率。**

## H200 上的文件入口

远端主机：`h200`。下列为远端绝对路径；模型权重未通过 fetch_reports.py 下载到本地。

| 项目 | 路径 |
| --- | --- |
| 代码仓库 | `/home/pengguanqi/Worksapce/Research/VLA-JEPA` |
| 专用 conda 环境 | `/home/pengguanqi/miniconda3/envs/vlajepa-piper` |
| 本次运行入口 | `/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/piper_press30hz_pose9_20260929` |
| 实际 scratch 存储 | `/data/scratch/pengguanqi/VLA-JEPA-runs/piper_press30hz_pose9_20260929` |
| 推荐检查点入口 | `/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/piper_press30hz_pose9_20260929/checkpoints/best/model.pt` |
| 最佳检查点实际目录 | `/data/scratch/pengguanqi/VLA-JEPA-runs/piper_press30hz_pose9_20260929/checkpoints/step_002000` |
| 最后检查点实际目录 | `/data/scratch/pengguanqi/VLA-JEPA-runs/piper_press30hz_pose9_20260929/checkpoints/step_002000` |
| 官方预训练权重 | `/data/scratch/pengguanqi/Models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt` |
| 已验证的数据集 | `/data/scratch/pengguanqi/Datasets/piper_elevator_lerobot_press_30hz` |

HOME 用户配额在第 400 步保存优化器时达到 100 GB 上限。运行文件逐项 SHA 验证后迁移到 scratch，原入口保留符号链接；从完整第 200 步检查点恢复并重算 201–400 步。旧日志尾部、旧第 400 步验证和未完成保存已归档，不计入当前有效训练历史。

## 可复核的来源

基础仓库 commit：`b49b016b53d5dd62e23036bd6e4a7b9481ae87b4`。数据集清单 SHA256：`c02eed880bd02ecb6fee4d594551c2c3e65fc871ad294178119c20d553cbaa1e`。

最佳模型 SHA256：`6793045b0b00e863737fe4c0742fde688472d415ab4d8a788ded6254ebe1d1f4`。最后模型 SHA256：`6793045b0b00e863737fe4c0742fde688472d415ab4d8a788ded6254ebe1d1f4`。

以下 SHA256 对应 fetch_reports.py 拉取的本地原始报告字节；大权重本身未由此报告脚本重新读取。远端主机与环境目录来自报告生成命令参数。

| 来源报告（相对 reports/） | SHA256 |
| --- | --- |
| `piper_setup/code_manifest.json` | `e9f8c79070f5202542a9be63c7ebb8bc201902ae1753ff7d26c58294b29d9c4b` |
| `piper_setup/dataset_integrity.json` | `763f6663bffda9fbc613c1736d3c4554c7d34b8bb19a6dd0cad3a0d893c08a27` |
| `piper_setup/environment_freeze.txt` | `bd4763a4e45b0e8e128055cf1c7eb88c04210e61e56b3b022177b5e19aebd32a` |
| `piper_setup/remote_tests.log` | `4e44e4910cd97d855172abe3876fcd3afa8309554dc91b987cafe6e303882252` |
| `runs/piper_press30hz_pose9_20260929/acceptance_summary.json` | `bb7000ed09376e8a137e142ca609d497f2b2aa6246ccfa2226af3b7b64a4de3f` |
| `runs/piper_press30hz_pose9_20260929/action_schema.json` | `30dec4dd334e087f82ff1f6c71e02703e6515698323597a99f0adbd6a23d8422` |
| `runs/piper_press30hz_pose9_20260929/checkpoints/best/manifest.json` | `ad4af144b76e7a5db24e8bd7459a98c04b5f4257be38166d82d9b0dad124fffc` |
| `runs/piper_press30hz_pose9_20260929/checkpoints/last/manifest.json` | `ad4af144b76e7a5db24e8bd7459a98c04b5f4257be38166d82d9b0dad124fffc` |
| `runs/piper_press30hz_pose9_20260929/completed.json` | `60e695654a9ff05b49661b4fa4c97f45077c51a60300f9ceca0b86918d18f765` |
| `runs/piper_press30hz_pose9_20260929/config.yaml` | `88c56c2c341d460278bf63c5bcf053880d9bc2d1e0b0f1b6016944d2f7c97665` |
| `runs/piper_press30hz_pose9_20260929/independent_inference.json` | `bd79b97f32c94423cb234fdbd1765e70ec9db5e472073c81bc516c397bbcf618` |
| `runs/piper_press30hz_pose9_20260929/parameter_update_evidence.json` | `4b1a50b129397cfa1e2d4fad4b72ff677953bdfdb5ba458e25c8259735b7b0a5` |
| `runs/piper_press30hz_pose9_20260929/parameters.json` | `56caab2e08768b0fc1b273e8e9537358c8f9297bd25a6da67fb8e1eba59514d8` |
| `runs/piper_press30hz_pose9_20260929/pretrained_loading.json` | `7243940404ac577c3e4e4afefb4a6fd6e76d65a17f2c1793a8e07e062b701a53` |
| `runs/piper_press30hz_pose9_20260929/recovery_archive/from_step_000200_attempt01/recovery_manifest.json` | `e4ac51c4610ab809cba2873bdd71802dee7067e8c1bbcb702c9a704018425da9` |
| `runs/piper_press30hz_pose9_20260929/split.json` | `ec19dd89203c5e8a9043050da9d7d05f70edcda9853886ac4031ef7c9644b774` |
| `runs/piper_press30hz_pose9_20260929/storage_migration.json` | `c4325601e9befa39712670392281af055b1b068ec20a73c11799084b3da1d31d` |
| `runs/piper_press30hz_pose9_20260929/train.jsonl` | `bc4d4f827448bebcf558a9cd2ff14d6bb07ee69de89b375d42925cb189e91301` |
| `runs/piper_press30hz_pose9_20260929/training_contract.json` | `65e92422fa6858046df63e452fdbb95748e61cb9c7e826213f23d709870ceb82` |
| `runs/piper_press30hz_pose9_20260929/validation_002000.json` | `c31b45ab8531491b72a60633a68b6501d22e68408083cf7734ac11305f8eb96c` |
| `runs/piper_press30hz_pose9_20260929/validation_anchors.json` | `b89fe0eafa6fd2ef70fd3c4ad9454a4a0bcb8633e523b597b8daea1b7839e0b0` |
