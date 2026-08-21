# MCGS-SLAM 论文实验运行清单

论文版本：arXiv:2509.14191v3（2026-03-09）

## 统一评测协议

- Waymo：Front、Front-Left、Front-Right 三个同步 RGB 相机，使用原始畸变图像。
- Oxford Spires：三个鱼眼相机；为适配 pinhole 模型，论文先进行去畸变。
- AirSim：四相机 aircraft rig。
- 外观指标：建图结束后，在所有关键帧上计算 PSNR、SSIM 和 LPIPS。
- 轨迹指标：与 GT 做 Sim(3) Umeyama 对齐后的 ATE RMSE（米）。
- Waymo ATE：论文分别测试启用/关闭 JDSA，并在表格中报告较好的一项。因此严格复现需要每个序列跑两次。
- 完整模型外观评测：MCBA + Metric3Dv2 depth prior + JDSA + offline global BA/3DGS refinement。

## Waymo Open Dataset

| 序列 | 论文 PSNR | 论文 SSIM | 论文 LPIPS | 论文 ATE (m) | 本机 MCGS 三相机数据 | 待跑配置 | 状态 |
|---|---:|---:|---:|---:|---|---|---|
| 100613 | 27.09 | 0.830 | 0.223 | 0.398 | 有，198 帧 | JDSA ON；JDSA OFF | 已完成两种配置；论文规则取优 ATE 为 0.686 m |
| 132384 | 26.26 | 0.826 | 0.284 | 1.242 | 缺 | JDSA ON；JDSA OFF | 阻塞：只有双-agent转换数据，缺论文三相机原始数据和标定 |
| 134763 | 27.20 | 0.813 | 0.233 | 1.107 | 缺 | JDSA ON；JDSA OFF | 阻塞：只有双-agent转换数据，缺论文三相机原始数据和标定 |
| 152706 | 28.45 | 0.797 | 0.330 | 2.554 | 缺 | JDSA ON；JDSA OFF | 阻塞：只有双-agent转换数据，缺论文三相机原始数据和标定 |
| 158686 | 21.91 | 0.682 | 0.547 | 0.612 | 缺 | JDSA ON；JDSA OFF | 阻塞：只有双-agent转换数据，缺论文三相机原始数据和标定 |
| 153495 | 26.48 | 0.813 | 0.231 | 1.180 | 缺 | JDSA ON；JDSA OFF | 阻塞：只有双-agent转换数据，缺论文三相机原始数据和标定 |
| 106762 | 27.70 | 0.819 | 0.262 | 2.366 | 缺 | JDSA ON；JDSA OFF | 阻塞：只有双-agent转换数据，缺论文三相机原始数据和标定 |
| 163453 | 26.92 | 0.829 | 0.234 | 0.927 | 缺 | JDSA ON；JDSA OFF | 阻塞：只有双-agent转换数据，缺论文三相机原始数据和标定 |
| **平均** | **26.50** | **0.801** | **0.293** | **1.298** | — | — | — |

Waymo 每个序列的严格复现至少包含：

1. `MCBA + prior depth + JDSA`：用于完整模型外观指标，并计算 ATE。
2. `MCBA + prior depth, no JDSA`：用于计算另一项 ATE。
3. 两个 ATE 中取较小值，与论文 Table II 对比；外观指标使用完整模型结果与 Table I 对比。

## Oxford Spires Dataset

论文仅报告 ATE；本机 `/home/shaoqi/datasets/OxfordSpiresMultiagent` 当前不存在，项目中的链接无有效目标。

| 场景 | 论文 ATE (m) | 本机数据 | 状态 |
|---|---:|---|---|
| Bodleian Library | 7.665 | 缺 | 阻塞：需三鱼眼原始序列、GT 和去畸变标定 |
| Blenheim Palace | 3.391 | 缺 | 阻塞：需三鱼眼原始序列、GT 和去畸变标定 |
| Christ Church College | 1.551 | 缺 | 阻塞：需三鱼眼原始序列、GT 和去畸变标定 |
| Observatory Quarter | 0.924 | 缺 | 阻塞：需三鱼眼原始序列、GT 和去畸变标定 |

## AirSim Dataset

| 场景 | 论文 PSNR | 论文 SSIM | 论文 LPIPS | 本机数据 | 状态 |
|---|---:|---:|---:|---|---|
| Garden | 29.36 | 0.879 | 0.126 | 缺 | 阻塞：需论文 MC-Airsim 四相机数据及标定 |
| Factory | 28.37 | 0.924 | 0.083 | 缺 | 阻塞：需论文 MC-Airsim 四相机数据及标定 |
| Village | 28.10 | 0.853 | 0.219 | 缺 | 阻塞：需论文 MC-Airsim 四相机数据及标定 |
| **平均** | **28.64** | **0.885** | **0.143** | — | — |

## 已完成的预检查

- `100613` 的 10 关键帧 smoke test 已完成：
  - ATE RMSE：0.059141 m
  - PSNR：31.5884
  - SSIM：0.9013
  - LPIPS：0.1129
- 该 smoke test 不能与论文完整序列指标直接比较。
- 当前 GitHub 仓库存在两个复现问题：
  1. ~~README 和 `calib/100613.yml` 重复列出第四路 `front_right`，与论文三相机协议不一致。~~ **这一条是误诊，2026-08-09 核实代码后更正，见下节"槽位约定核实"。README 和 calib 都是对的；真正的问题在于本机复现跑没有按 README 传 4 个目录。**
  2. Eigen 未作为有效 Git submodule 提交，且 `setup.py` 的 sm_60/sm_61 编译参数与当前 Eigen 的 sm_70+ 要求冲突。

## 槽位约定核实：第 4 路不是冗余，本机复现跑实际只用了 2 路相机（2026-08-09）

逐行核实代码后确认，`--imagedir` 的槽位约定是 DROID-SLAM 双目设计的遗留：

- `mcgs_slam/motion_filter.py::track()` 第 113-114 行无条件执行 `indices = list(range(len(image)))` / `del indices[1]`。**槽位 1 是"双目右目"**：它的相关性特征图 `gmap`（第 123 行对全部槽位计算）保留下来供立体匹配使用，但它的图像/深度/法向/上下文特征都被排除在 `video.append()` 之外，**永远不会进入建图**。
- `mcgs_slam/options.py:48`：`args.multi = len(args.imagedir)`——**由命令行目录个数决定，不是由 calib 行数决定**。
- 因此 N 个 `--imagedir` 目录 ⇒ 实际参与建图的相机数 = **N - 1**。

按这个约定复核 `calib/100613.yml`：4 行内参 `[front, front_right, front_left, front_right]`，其中第 1 行（front_right）正是槽位 1 的双目右目占位；`baseline` 的值与 `T_cami_cam0` 里 front_right 那一行完全相同，进一步印证槽位 1 就是 front_right 作为 front 的立体伙伴。**去掉被丢弃的槽位 1 后，实际建图相机 = front + front_left + front_right = 恰好是论文的三相机协议。README 第 85 行写的 4 个目录是正确的用法，不是笔误。**

**但本机已完成的复现跑没有照做。** 证据（`output/paper_repro/waymo_100613_jdsa_on/`）：

| 证据 | 观察值 | 含义 |
|---|---|---|
| 点云文件 | 只有 `pc_1.ply`、`pc_2.ply` | `save_pc` 循环 `range(1, args.multi-1)`，只输出 2 个 ⇒ `args.multi=3` |
| 渲染目录 | 只有 `cam0/`、`cam1/` | GS 后端只收到 2 路相机 |
| 对照：全景跑（5 槽位） | `pc_1..pc_4` + 4 个 cam 目录 | 同一约定下 5 个目录 ⇒ 4 路建图，自洽 |

**结论：`paper_ablation_results.md` 里那三组消融数据（A/B/C）是在"只有 2 路建图相机"（front + front_left，front_right 因为占了槽位 1 被丢弃）的条件下跑出来的，不是论文的三相机协议。** 这很可能是 ATE 比论文差 72.5%（0.686 m vs 0.398 m）的一个重要原因——少了一路相机就少了一整组 rig 约束。外观指标（PSNR/SSIM/LPIPS）反而超过论文，也与"评测的相机更少、且都是覆盖较好的前向视角"这一点不矛盾。

**尚未验证**：以上是从产物文件反推的（当时的运行命令没有留日志），无法 100% 确认当时具体传了哪 3 个目录（推测是 `front front_right front_left`）。**修正实验（按 README 传 4 个目录重跑 A/B/C 三组）尚未执行**——2026-08-09 尝试时远程 ashton 的 GPU 被其他任务占满（24017/24576 MiB，100% 利用率），按本项目历史经验（`panoramic_4dgs_status.md` §3.3/§3.17 都有跑到一半被抢占导致结果作废的先例）没有强行启动，等 GPU 空闲后再跑。

## 已完成的完整序列结果

### Waymo 100613（198 帧，47 个关键帧，三相机）

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ATE RMSE (m) ↓ |
|---|---:|---:|---:|---:|
| 论文 | 27.09 | 0.830 | 0.223 | 0.398 |
| 本机：prior depth + JDSA ON | 27.5051 | 0.8488 | 0.1815 | 0.6864 |
| 本机：prior depth + JDSA OFF | 27.6265 | 0.8528 | 0.1740 | 0.7177 |

按论文 ATE 取优规则，本机结果为 0.6864 m：

- 外观三项达到论文值：PSNR +0.4151 dB，SSIM +0.0188，LPIPS 改善 0.0415。
- ATE 未达到论文值：比论文高 0.2884 m（误差约高 72.5%）。
- 输出目录：
  - `output/paper_repro/waymo_100613_jdsa_on`
  - `output/paper_repro/waymo_100613_jdsa_off`
