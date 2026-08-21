# 项目状态总览（入口文档）

**最后更新：2026-08-18（§3.41 全景 4DGS 跑通之后）**

本仓库同时进行着**两条独立的研究线**，共 5 份细节文档。本文件是唯一入口，只做导航与状态判定，**不复制细节**——所有数字与论证都在下面链接的文档里。

---

## 一、文档地图

| 文档 | 作用 | 什么时候看它 |
|---|---|---|
| **本文件** | 入口/状态判定 | 先看这个 |
| `panoramic_4dgs_plan.md` | 全景线的**研发计划**（阶段、判据、明确不做的事） | 想知道接下来做什么 |
| `panoramic_support_feasibility.md` | 全景线的**设计文档** | 想了解为什么这么设计 |
| `panoramic_4dgs_status.md` | 全景线的**完整实验日志**（§1–§3.26，673 行，按时间顺序，含多处就地更正） | 追某个具体结论的来龙去脉 |
| `panoramic_4dgs_test_report.md` | 全景线的结构化摘要 | ⚠️ **已过时**，只覆盖到 §3.19 前后，未包含 §3.20–§3.26 |
| `paper_experiment_list.md` | 论文复现的**运行清单**与阻塞项 | 想知道还差哪些序列/数据 |
| `paper_ablation_results.md` | 论文复现的**消融结果** | ⚠️ 引用数字前先读它开头的两条保留 |

---

## 二、线一：全景 4DGS（本项目的主要研究目标）

### 🚀 怎么跑（当前生产路径，一条命令）

```bash
python demo.py --calib calib/equirect_6face.yml \
               --imagedir data/synth_erp_room_dynamic_fast_hires \
               --config config/config_gaussian_4d.yaml \
               --stride 1 --rgbd \
               --output output/pano4d
```

环境是 `magsslam`（**不是** README 里写的 `mcgs_slam_v1`，那个不存在）。单次约 8 分钟（RTX 3090）。

产物：

| 路径 | 内容 |
|---|---|
| `renders/erp_panorama/` | 观测时刻的 360° 全景（99.9% 球面） |
| `renders/erp_panorama_interp/` | **从未被观测过的时刻**的 360° 全景 |
| `renders/time_sweep/` | 单面时间扫掠，49 帧含 36 个未观测时刻 |
| `4dgs_final.pt` | 每高斯带时间中心/半径/速度的 4D 地图 |

**验收（会逐条检查"全景 4DGS"四个字，不是只看均值）**：

```bash
python tools/export_synthetic_gt_poses.py --sequence data/synth_erp_room_dynamic_fast_hires
python tools/verify_panoramic_4dgs.py --run output/pano4d \
       --sequence data/synth_erp_room_dynamic_fast_hires \
       --gt-poses data/synth_erp_room_dynamic_fast_hires_gt_poses.txt
```

当前生产配置在这 9 项上全部 PASS。工具本身也验证过会 FAIL：对 §3.49 那个坏跑报「帧间极差 11.52 dB」，对 4 面跑报「覆盖 48.1%」+「无插值全景」。

### 做到了

- **全景 cubemap 支持（P1）**：ERP→cubemap 分解、复用多相机 rig 管线、分辨率适配检查，实现完整、测试充分。**已在真实拍摄的 ERP 照片上跑通**（360VOTS，30000 步零崩溃，见 §3.22）。这是本项目最扎实的一块。
- **DeformNet 在普通 pinhole 数据上的独立验证（P4a）**：实现并测试通过。

### ✅ 2026-08-18：全景 4DGS 已跑通（§3.41）

四个字都有实现且验证过：**全景**输入（ERP→cubemap，真实拍摄数据上跑通过）、**4D** 表示（每个高斯带时间中心/时间半径/速度）、**未观测时刻可渲染**（时间扫掠 36/36 帧物体都在，逐时刻 bank 是 0）、**速度优于基线**（0.746 对 0.806）。

配置：`config/config_gaussian_4d.yaml`；速度来自几何配准而非训练（§3.32–3.36 证明光度损失学不出物体运动）。

### ✅ 2026-08-19：真正的全景 4DGS 已交付（§3.47）

6 面 cubemap（`calib/equirect_6face.yml`）+ 未观测时刻的全球面全景，n=3：

| 指标 | 4 面 | 6 面 | 差 |
|---|---:|---:|---:|
| 面域 PSNR（仅共有 4 面） | 25.39 ± 0.40 | **28.92 ± 0.03** | +3.5 dB |
| **ERP `full` PSNR** | 8.71 | **25.80** | **+17.1 dB** |
| 球面覆盖 | 46.5% | **99.9%** | — |

**§3.50 起生产配置为双半径**（`time_scale_init: 0.25` + `interp_time_scale: 0.5`）：ERP covered **27.38 ± 0.21**，帧间极差从 10.04 降到 **0.73**——t=1 那个 11 dB 的坏帧已消除，且插值帧的物体反而比之前更实。

产物：`renders/erp_panorama_interp/` —— 8 张 99.9% 球面覆盖、时刻从未被观测过的全景图。

⚠️ **引用渲染数字前先确认域**：`final_result_kf.json` 的 `per_cam` 是 **cubemap 面**（pinhole 域，畸变已被定义掉），不能用来论证全景/抗畸变。ERP 域的数字用 `tools/eval_erp_panorama.py` 单独算，且 `covered` 与 `full` 是两个不同的问题。

**当前最大的三个短板**（按杠杆排序）：
1. **速度估计**离上界仍有 **38%** 空间（生产 0.957 对 v=gt 0.595，§3.48）。生产估计器已于 §3.48 从 ICP 换成质心差分（0.975 → 0.957，n=3 对 n=3 取值范围不相交），但那只吃掉差距的 5%；剩下的是"表面片质心 ≠ 物体中心"的结构性偏差，两种估计器都基于 bank 几何中心，**换配准算法解决不了,得换信息源**
2. **物体放置精度**：根因是"表面片质心≠物体中心"的结构性偏差，需物体级先验（§3.40）。6 面让渲染变好但放置略变差（0.906 → 0.975），原因未查
3. **真实动态全景数据**仍空缺（§3.21/§3.30）

⚠️ **§3.40 的"深度是最大杠杆"已被 §3.43 推翻**：那 12.4 dB 是 metric3d 在这个合成房间上 OOD 崩溃造成的伪影（结构相关性为负），不能外推到真实数据。合成实验一律用 `--rgbd`。

### 2026-08-15 更新：能插值的表示已接通，瓶颈换了地方

下面"没做到"一节记录的是 8-10 暂停时的状态，其中"没有能插值的表示"这一条**已经不成立**——§3.27–§3.29 把物体级 SE(3) 轨迹修到可训练、接进 `GaussianModel`、并接上了渲染入口。当前状态：

- **观测时刻零回归**：可见性与逐时刻 bank 逐位相同，位置位移精确为 0（§3.29）
- **未观测时刻能渲染出物体**：bank 显示 0 个动态高斯的时刻，新路径显示 69~2404 个（§3.29）
- **新瓶颈：12 个动态 bank 里 6 个被离群高斯污染**（跨度 27~102 世界单位 vs 干净的 0.3）。两侧关键点都干净时物体放置误差 0.42~0.52（小于球半径 0.6），否则 21~122。离群点 opacity 正常、scale 偏大、集中在非 cam0 的 cubemap 面——**这个问题早于本次改动存在**（§3.29）
- **轨迹关键点目前拿不到梯度**（训练只渲染观测时刻，而在该时刻搬运是恒等），要监督它必须有观测时刻之间的损失（§3.29）
- **⚠️ 重建世界系与合成器房间系差约 10.5 倍尺度**，任何与 GT 位置的比较必须先做相似变换对齐（§3.29）

### 没做到（8-10 暂停时的记录，插值一条已被 §3.29 推翻）

**当时还没有一个够得上叫"4D"的表示。**

- **原设计的全局 `(xyz,t)→delta` 形变场：已证伪并关闭。** 去掉 ROI 作弊通道后的干净实验证明它学不会物体运动（动态区域 PSNR 11.82，反而低于静态基线 12.17）。此前看似有效的版本是网络在放大/重绘 canonical Gaussian 作弊。见 §3.19。
- **替代方案"显式时间切片动态层"：有正向信号，但不是 4D 表示。** 20 次独立跑确认优于静态基线（18/20 为正，统计显著），但：
  - **不能插值到未观测时刻**——本质是每个时刻存一份独立点云的查找表
  - **依赖 oracle 颜色标签**——真实数据上不存在
  - 效应量 0.65~0.81 dB，而噪声标准差 0.46~0.59 dB，信号仅略高于噪声
  - 见 §3.19、§3.20、§3.24

### 卡在哪

1. **没有合适的真实数据**。两轮调研全部落空：Princeton365（发布的是 pinhole 裁剪视图非全景）、360VOTS（真 equirect 但**相机固定**，测不了 SLAM）、JRDB（相机会动但**无位姿 GT**）、TBD Pedestrian Set 2（唯一同时满足条件的候选，**格式待人工确认**，见 §3.21）。所有结论目前都建立在同一个合成房间上。
2. **表示本身缺插值能力**——这是最根本的一条，见下方"下一步"。

### 下一步（2026-08-16 起）

§3.30 换了第四条表示路线（原生 4D 高斯 / 时间切片，见上），机制成立，但同时**发现了一个更靠前的阻塞**：

**⚠️ 合成序列里的物体几乎不动——每帧位移只有球直径的 22%，相邻帧重叠 78%。** 后果是 v=0（只把时间维展宽）已经打平用真实运动的上界 v=gt（0.263 vs 0.259），速度项无事可做；这大概率也是 §3.19/3.20 里动态增益只有 0.65~0.81 dB 而噪声就有 0.46~0.59 的原因。**在换数据之前，任何 4D 表示的优劣在这个序列上都测不出来。**

按性价比排序：

1. **把合成物体的运动幅度调大约 5 倍**（`tools/make_synthetic_erp_room.py` 已有三种运动 profile，改幅度即可）——最便宜的解锁，不必等真实数据，同时给动态收益一个不被噪声淹没的量程
2. **把 `time_scale`/`velocity` 接进 `GaussianModel` 并训练**（照 §3.28 `dynamic_object_id` 的 15 个传播点做），这是让速度被"学出来"而非"估出来"的必要步骤
3. **动态 bank 的离群污染**（12 个里 6 个）：离群点集中在非 cam0 的 cubemap 面且 scale 偏大，与 §3.11/§3.14 的面间质量差异同向
4. **换直接 ERP 光栅化**：本地 GGPS/PanoLOG 已有可用的 OmniGS LonLat CUDA 实现（前向反向齐全，接口只差 `camera_type=3`），能一次性消掉第 3 条那一整类面间问题；代价是要合并两个 rasterizer 派生，且 **OmniGS 是 GPL**，需先想清楚许可

原"先做表示，再找数据"的判断需要修正为：**表示已经有三个候选可用，现在卡住的是数据能不能把它们区分开**。真实数据调研见下方"卡在哪"，另有 §3.30 记录的新候选（360DVO、OmniLocalRF 数据集等）。

---

## 三、线二：论文复现（arXiv:2509.14191v3）

### 关键发现

**此前的消融实验一直只用了 2 路建图相机，而非论文的 3 路。** 根因是一个此前无人注意的约定：`motion_filter` 无条件丢弃槽位 1（DROID 双目右目占位），且 `args.multi = len(args.imagedir)`，所以 **N 个 `--imagedir` 目录 ⇒ N−1 路建图相机**。README 要求传 4 个目录才是论文的三相机配置。

补齐第三路后 ATE 大幅改善（A 组与论文差距 +81.7% → +7.1%）。详见 `paper_ablation_results.md`。

**判断相机数的三个快速方法**：日志 `PhysicalWindow: active_views=M` ÷ 物理时刻数、`pc_*.ply` 文件个数、`renders/image_after_opt/cam*` 目录个数。

### 已确立的方法论

**ATE 是确定性的（n=1 即可），外观指标不是（需 n≥10）。** 三次独立重跑给出逐字节相同的 `traj_mcgs.txt` 而 PSNR 各不相同——因为轨迹来自 DROID 跟踪层的 `video.globuf`，与非确定性的 GS 精修阶段无关。

对应地，本项目在小样本上栽过两次（n=1 的"退化"、n=3 的"加密更差"），补足 n=10 后**两次结论都被推翻**。

### 未决

- **ATE 绝对值不可引用**（见 `paper_ablation_results.md` 开头保留）
- **JDSA 在多相机下有害**，建议保持关闭。已定位两条独立致害通路（`dscales` 规范漂移约 30%、视差更新约 70%），**均非索引问题**，无一行式修复
- 多数序列因缺原始多相机数据而阻塞，本机仅 `100613` 可跑

---

## 四、已知 bug 清单（按严重度）

| 严重度 | 位置 | 问题 | 状态 |
|---|---|---|---|
| 高 | `depth_video.py::rm_keyframe` | 移位时不维护 `kf_stamps`，而它参与关键帧选取判据 → 改变轨迹本身 | **修复正确但未应用**：会改变关键帧选取，单条序列无法评估好坏 |
| 中 | `eval_utils.py::eval_rendering` | 把帧序号当归一化物理时间戳传，oracle 模式下会把动态高斯全部置零 | 未修（**报数用的是 `eval_rendering_kf`，那条路正确**） |
| ~~中~~ | `object_trajectory.py::add_observation` | 重绑 `nn.Parameter`，使已建的优化器失效 | **已修（§3.27）** |
| ~~中~~ | `object_trajectory.py::quaternion_slerp` | `torch.where` 双分支回传，近似相同旋转处产生 NaN 梯度 | **已修（§3.27）** |
| 中 | 动态 bank 离群污染 | 12 个 bank 里 6 个跨度达 27~102 世界单位，毁掉物体放置 | 未修（§3.29，当前主瓶颈） |
| 低 | `object_trajectory.py::add_observation` | `optimizer=None` 仍是默认值，忘传的调用点会静默不训练 | 未修（§3.28 记录；§3.29 的调用点是有意传 None） |
| 中 | `gs_backend.py:647` | replay pool 每次迭代从全历史重建，开销随运行时长线性增长 | 未修（性能，非正确性） |
| 低 | `droid_frontend.py` | `args.multi<=2` 分支传了 `FactorGraph.update()` 不接受的 `use_scaling` | 未修（靠永远用 ≥3 槽位绕开） |
| 低 | 其余 5 项 | 见代码审查记录 | 未修 |

---

## 五、不要重走的弯路（已证伪，有实验记录）

- **DeformNet 的超参数调优**：正则权重、warmup、傅里叶频率、学习率、输出上限、架构解耦，六个角度全部无效（§3.5–§3.9）
- **动态采样密度调优**：1/8 vs 1/4，n=10×2 后仍无法区分（§3.24）
- **训练窗口调度改造**：前提不成立——`color_refinement`（决定最终指标的 30000 步）本就是均匀随机采样，不存在窗口稀释（§3.25）
- **JDSA 的索引配对"修正"**：索引错配是真的，但改"对"反而让 ATE 差 5 倍（`paper_ablation_results.md`）
- **单独应用 `kf_stamps` 修复**：见上表

---

## 六、环境与运行状态

- **远程 GPU**：`ssh ashton`（RTX 3090，共享机器，被抢占废掉过至少 3 次实验）。repo 在 `/home/shaoqi/Project/MCGS-SLAM`，conda 环境 **`magsslam`**（不是名字相近的 `mcgs_slam_v1`）
- **新服务器**：`barklaviz1.liv.ac.uk`（2× NVIDIA L4，各 23 GB），需 keyboard-interactive 认证 + ControlMaster 复用。**本次暂停时未使用**
- **本地 Mac**：无 GPU，仅改代码
- **git**：本地与 ashton 内容一致、均无未提交改动，领先 `origin/main` 37 个 commit，**未 push**
- **数据质量**：已全面审计，干净（Waymo 5 相机各 198 帧且逐一对齐、合成数据集分辨率符合文档、GT 轨迹点数匹配、360VOTS 图/掩码零缺漏）
### 2026-08-18 输出目录清理

磁盘到 91%（另一个项目要用机器），删掉了本次会话里**失败 / 中间验证 / 被取代**的输出，约 **3.5 GB**。**所有数字都已在 `panoramic_4dgs_status.md` 记录，删的只是可重跑的产物**：

| 删除的目录 | 类别 | 结论记在哪 |
|---|---|---|
| `loo_gaussian4d_r1~r3` | 失败（leave-one-out 反而更差） | §3.33 |
| `pool_ts1_r1~r3` | 失败（加大时间半径使速度学成反向） | §3.35 |
| `dense4_r1`、`dense2_r1` | 失败（采样密度 4× 无改善） | §3.39 |
| `step1_buffer_check`、`step2_optimizer_check`、`step3_delta_check` | 中间验证，各自检查点已通过 | §3.32 |
| `step4_gaussian4d_r2~r5` | n=5 重复，均值/σ 已记录（保留 `r1`） | §3.32 |
| `fast_oracle_timeslice_r2~r5` | n=5 重复，均值/σ 已记录（保留第一个） | §3.31 |
| `pool_gaussian4d_r1~r3` | 池化机制已成为默认，`pano4dgs_v2` 即池化产物 | §3.34 |
| `pano4dgs_v1` | 被 `v2` 取代 | §3.41 |
| `static_pano_gt`、`static_pano_gt2` | 含已修复的红蓝通道/曝光 bug，被 `gt3` 取代 | §3.44 |

**保留的关键产物**（文档中的数字直接来自它们，删了就无法复核）：

| 目录 | 为什么留 |
|---|---|
| `step4_gaussian4d_r1` | §3.32/3.36/3.40 大量数字的来源（v=icp、方向余弦、面间分析） |
| `gtdepth_r1` | §3.40 的 +12.4 dB 证据 |
| `pano4dgs_v2` | §3.41 最终成果（time_sweep / ERP 全景产物），已导出 |
| `fast_oracle_timeslice` | §3.31 基线 |
| `objid_smoke2` | §3.28/3.29 产物 |
| `static_mono`、`static_gtdep` | §3.42 静态基线对照 |
| `static_pano_gt3` | §3.44 修复后的 ERP 全景结果 |

- **⚠️ ashton 上有 10 个不完整的输出目录**（文档中已记录为失效的跑，含 1.6G 的 `bugfix_check`），**不要当作有效结果引用**：
  `bugfix_check`、`deform_4cam_control`、`deform_5slot_control`、`synth_erp_deform_diag`、`synth_erp_deform_final_lowlr`、`synth_erp_deform_lowlr`、`synth_erp_deform_smoke`、`synth_erp_dynamic_hires_final`、`synth_erp_dynamic_hires_oracle_gate`、`synth_erp_dynamic_hires_rampfix_rawreg`
