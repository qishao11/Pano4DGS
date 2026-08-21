# 全景相机接入 MCGS-SLAM 可行性与接口设计文档

版本：草稿 v1（基于 commit `d1410c1`，分支 `main`）
范围：只评估**全景(360°/等距柱状)相机**的移植难度与接口设计。4DGS 作为后续独立工作，本文档只标出预留接口点，不展开方案。

---

## 0. 结论摘要

- 当前代码库里**没有任何全景相机代码**（不是没写完，是完全没有），也没有 4DGS 代码。全仓库是标准 **pinhole 多相机刚性阵列 + DROID-SLAM 跟踪 + 静态 3D Gaussian Splatting** 架构。
- 好消息：DROID-SLAM 的跟踪/光流关联/BA 部分，相机模型相关的数学**几乎全在 Python/PyTorch 层**（`projective_ops.py`、`geom/ba.py`），CUDA 关联核（`correlation_kernels.cu`）本身只认像素坐标网格，不关心这坐标是怎么投影出来的。这意味着跟踪侧的全景适配**不需要碰 CUDA**，只需要新增一套等距柱状（equirectangular）版本的投影/反投影函数。
- 坏消息：渲染侧完全不同。`diff-gaussian-rasterization` 的 CUDA 光栅化器是围绕**线性投影矩阵 + tanfovx/tanfovy 的透视 NDC 假设**写死的（`GaussianRasterizationSettings`），360° FOV 在这个假设下不成立，必须动 CUDA 或绕过它。这是全景改造里工作量和风险最大的一块。
- 现有的"多相机"支持（`cam_idx`、`T_cami_cam0`、每路独立 pinhole 标定）跟全景无关，但**结构上可以复用**：把一个全景相机拆成 N 个虚拟 pinhole 视角（cubemap/多面体分解）后，正好可以套进现有的多相机 rig pipeline，几乎不用改跟踪和渲染代码。这是本文档推荐的低风险落地路径（见第 5 节）。

---

## 1. 现状：当前 pinhole 流程总览

```
标定 YAML (calib/*.yml, camera: 'pinhole', [fx,fy,cx,cy,k1,k2,p1,p2])
   │  mcgs_slam/options.py: load_configs()
   ▼
streams.py: image_stream()  ── cv2.undistort() 假设 pinhole+径向/切向畸变
   │  yields (t, images[N,3,H,W], intrinsics[N,8], timestamp)
   ▼
mcgs.py: Mcgs.track()
   │
   ├─▶ motion_filter.py → droid_frontend.py (DROID-SLAM 光流跟踪/关键帧选取)
   │        │  核心投影数学：geom/projective_ops.py:
   │        │    iproj()  第19-38行  "pinhole camera inverse projection"
   │        │    proj()   第40-66行  "pinhole camera projection"
   │        │  BA/因子图：geom/ba.py（纯 Python/PyTorch，消费 proj/iproj 的输出与雅可比）
   │        │  关联体采样：src/correlation_kernels.cu（CUDA，只认 2D 像素坐标，不关心相机模型）
   │
   ▼
mcgs.py: Mcgs.call_gs() → 打包 poses/images/depths/intrinsics/cam_idx
   ▼
gs_backend.py: GSBackEnd.process_track_data()
   │  getProjectionMatrix2(fx,fy,cx,cy,...) → 透视投影矩阵（graphics_utils.py:72-93）
   │  Camera.init_from_tracking(...)  camera_utils.py:65-85
   │      注意：tstamp 已经作为字段存在 cam.tstamp，但 Gaussian 侧从未使用
   ▼
GaussianModel.extend_from_pcd_seq() → create_pcd_from_image_and_depth()
   │  gaussian_model.py:113-183
   │  用 o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy) 把新关键帧的 RGBD 反投影成点云种子新 Gaussian
   ▼
render() (gaussian/renderer/__init__.py:17-85)
   │  tanfovx = tan(FoVx/2), tanfovy = tan(FoVy/2)   ← 透视相机专属概念
   │  GaussianRasterizationSettings(viewmatrix, projmatrix, tanfovx, tanfovy, ...)
   ▼
diff-gaussian-rasterization (CUDA, thirdparty/diff-gaussian-rasterization/)
   │  cuda_rasterizer/forward.cu 等：基于线性投影矩阵 + NDC 的 tile-based 光栅化
   ▼
渲染图像 / 深度 / 用于 3DGS 优化的梯度
```

---

## 2. 全景相机改造点逐段分析

### 2.1 数据接入层 —— 难度：低

- `mcgs_slam/streams.py:image_stream()` 第 40-42 行：`cv2.undistort(image, K, calib[i][4:])`，假设 pinhole + 径向/切向畸变模型。
- 全景图（等距柱状全景图，ERP）不需要"去畸变"这一步，直接是完整的 360°×180° 图像。改造点：
  - `calib/*.yml` 新增 `camera: 'equirect'` 分支（`options.py:load_configs()` 第 52-53 行目前读了这个字段但从没 dispatch 过，改造成本很低）。
  - `streams.py` 按 `args.camera` 分支跳过 `cv2.undistort`，直接 resize/裁切。
- 结论：改动量小，风险低。

### 2.2 跟踪 / 光流 / BA 层 —— 难度：中，但集中在 Python 层

这是 SLAM 几何精度的核心，也是本次评估里**唯一需要重新推导数学**的地方，但好消息是全部在 Python/PyTorch 里，不涉及 CUDA 改动：

- `mcgs_slam/geom/projective_ops.py`：
  - `iproj()`（第19-38行）：像素 → 归一化相机系射线，pinhole 公式 `X=(x-cx)/fx, Y=(y-cy)/fy`。全景需要替换成球面反投影：`(x,y)` 归一化到经纬度 `(θ,φ)`，再转 3D 单位射线方向。
  - `proj()`（第40-66行）：3D 点 → 像素坐标，pinhole 公式 `x=fx·X/Z+cx`。全景需要替换成球面正向投影 `θ=atan2(X,Z), φ=asin(Y/r)` 再映射回 ERP 像素坐标，并重新推导对应的雅可比（`proj_jac`，第58-62行，供 BA 用）。
  - `actp()`（第68行起）：位姿作用于点云，是纯 SE3/Sim3 变换，与相机模型无关，**不需要改**。
- `mcgs_slam/geom/ba.py`：`BA()`/`MoBA()`/`JDSA()` 等函数只消费 `proj`/`iproj` 返回的坐标和雅可比做高斯-牛顿/LM 优化，本身不含相机模型假设，**理论上不用改**，只要上游喂给它的雅可比是正确的球面雅可比。
- `src/correlation_kernels.cu`：`corr_index_forward_kernel`（第20行起）只对一张 `[N,C,H,W]` 特征图按给定的 `coords[n][0/1][y][x]` 做双线性采样，**完全不关心这个坐标是从 pinhole 还是球面投影算出来的**。CUDA 关联核**不需要改**。
- 已知风险点（这部分是真正的技术难度，不是工程量问题）：
  1. **极区畸变**：等距柱状投影在南北极附近像素密度趋于无穷/为零，光流估计和特征关联在极区会严重退化，可能需要专门的采样策略（如立方体贴图局部块，见 2.4 节的路线对比）。
  2. **雅可比推导正确性**：BA 收敛依赖精确的解析雅可比，球面投影的雅可比比 pinhole 复杂（涉及三角函数），需要仔细推导并做数值梯度检验。
  3. **DROID 预训练权重**（`pretrained_models/droid.pth`）是在 pinhole 图像上训练的光流/关联网络，全景图像的像素邻域语义（尤其是横向环绕、纵向极区压缩）跟 pinhole 差异很大，直接复用权重可能精度下降，**大概率需要微调或重新训练关联网络**，这是比数学公式本身更大的不确定性来源。

### 2.3 Gaussian 初始化 / 种子化 —— 难度：低到中

- `gaussian_model.py:create_pcd_from_image_and_depth()`（第113-183行）用 `o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)`（第133-140行）把新关键帧的 RGBD 反投影成世界系点云，作为新增 Gaussian 的初始位置/颜色。
- 改造点：全景关键帧无法用 Open3D 的 pinhole 反投影，需要自己写一个球面反投影函数（等距柱状像素 → 经纬度 → 3D 射线 → 乘以深度值 → 世界系点），绕开 `o3d.camera.PinholeCameraIntrinsic` 直接手写 numpy/torch 版本。这部分逻辑不复杂（跟 2.2 的 `iproj` 是同一套数学），主要是替换掉 Open3D 依赖。

### 2.4 渲染层 —— 难度：高（本次改造的核心风险与工作量所在）

- `gaussian/renderer/__init__.py:render()` 第25-26行：`tanfovx = tan(FoVx*0.5)`，`camera_utils.py` 里 `Camera.FoVx/FoVy` 通过 `focal2fov(fx, W)` 算出——**这套 FoV 概念对 360° 全景图完全不成立**（全景没有一个"视场角"，是全向的）。
- `thirdparty/diff-gaussian-rasterization/` 的 CUDA 光栅化器（`cuda_rasterizer/forward.cu` 等）内部按 `viewmatrix` + `projmatrix`（透视投影 + NDC 变换）把每个 3D Gaussian 投到屏幕空间，再做 tile-based 排序和 alpha blending。这套投影数学**写死了透视模型**，没有球面投影的分支。

有两条可行路线，工作量和风险差异很大：

| 路线 | 做法 | 优点 | 缺点 | 工作量/风险 |
|---|---|---|---|---|
| **A：Cubemap / 多面虚拟 pinhole 分解**（推荐先做） | 把每个全景关键帧渲染/监督拆成 6 个（或更少，如 4 个水平面）虚拟 pinhole 子相机，每个子相机 FoV ~90°，直接复用现有 `Camera`/`render()`/CUDA 光栅化器，跟现有"多相机 rig"的 `cam_idx` 机制天然契合 | 不需要碰 CUDA 光栅化器；风险和工作量可控；可以复用现有的 `T_cami_cam0` 式外参管理 | 立方体贴图的接缝处可能有细微的几何/光度不连续；每帧要渲染 N 个子视角，训练/推理开销乘 N | 中 |
| **B：原生等距柱状光线追踪渲染器** | 改 CUDA 光栅化器（或写一个新的），对每个 ERP 像素生成一条球面射线，做基于射线-高斯相交/积分的渲染（不再是屏幕空间 tile 光栅化范式），需要重新推导 Gaussian 的球面投影协方差 | 无缝、理论上更精确，是学术界（如 OmniGS 一类工作）验证过的方向 | CUDA 开发量大，需要重写协方差投影、tile 划分策略（球面上没有均匀的屏幕空间网格），调试周期长，是整个改造里最大的不确定项 | 高 |

**建议**：先走路线 A 验证端到端可行性和精度，路线 B 作为后续如果对拼接质量/性能要求更高时再投入。

---

## 3. 接口设计建议（具体到代码改动点）

按照路线 A（cubemap 分解）给出最小接口改动集合：

1. `calib/*.yml`：新增 `camera: 'equirect'`，以及全景专属参数（分辨率、是否只用水平 4 面等）。
2. `mcgs_slam/options.py`：`load_configs()` 按 `args.camera` 分支加载不同的相机参数结构；新增 `--camera_model` 或复用 `camera` 字段做 dispatch（当前该字段读了但完全没用，第52-53行）。
3. 新增 `mcgs_slam/geom/spherical_ops.py`（对标 `projective_ops.py`）：实现 `equirect_iproj()` / `equirect_proj()` 及其雅可比，供跟踪/BA 复用；跟踪主循环按 `args.camera` 分支调用 pinhole 版或 spherical 版。
4. 新增 `mcgs_slam/cubemap.py`：全景帧 → N 张虚拟 pinhole 子图（含每个子相机的虚拟内参/外参生成），子图直接喂给现有 `image_stream`/`Camera.init_from_tracking` 流程，复用现有的 `cam_idx` 多相机机制（`gs_backend.py`、`mcgs.py:call_gs()`），**这一层之后的代码基本不用改**。
5. `gaussian_model.py:create_pcd_from_image_and_depth()`：如果走cubemap路线，子图仍是 pinhole，这一步**不需要改**；只有走路线 B 才需要新的球面反投影函数。
6. 关键判断点：DROID 光流/关联网络要不要在等距柱状全帧上直接跑（需要 2.2 节的 spherical_ops），还是也拆成子图跑跟踪（完全复用现有 pinhole 跟踪，代价是子图边界的跟踪连续性要额外处理）——**这是需要你决策的架构选择**，直接影响 2.2 节工作量是否会被触发。

---

## 4. 4DGS 设计方案

### 4.1 现有架构给出的两个关键约束

读完 `gs_backend.py` 全文后，有两点直接决定了 4DGS 该怎么设计，而不是照搬离线 4DGS 论文的做法：

1. **两阶段结构天然对应"在线局部" + "离线全局"**：
   - 在线阶段：`process_track_data()`（第72-113行）每来一个新关键帧就调用 `initialize_map()`（211-249行，仅首帧）或 `map()`（251-319行，iters=10），只用当前滑动窗口 `current_window`（容量约10帧，见85行）+ 2个随机历史帧做优化，是**因果的、只看得到"到目前为止"的数据**。
   - 离线阶段：`finalize()`（178-189行）调用 `color_refinement()`（321-376行），在全部收集到的关键帧集合 `self.viewpoints`（此时序列已经跑完，未来时刻也能看到）上做 `max_steps`（配置里是 30000，我们冒烟测试日志里看到的 "Global GS Refinement...30000" 就是这一步）迭代的全局精修，同时优化 `cam_rot_delta/cam_trans_delta`（位姿残差）和 `exposure_a/b`。
   - 这正好对应大部分离线 4DGS 论文"看得到整段视频"的假设——**只有 `color_refinement()` 这个离线阶段才适合做真正意义上的全局时间一致性优化**；在线阶段的形变场只能是因果的、局部的，且天然存在"看不到未来/容易遗忘早期时刻"的风险。
2. **`process_global_track_data()` 会直接原地改写 Gaussian 参数**（120-137行）：全局 BA 修正位姿/尺度后，直接对 `self.gaussians._xyz`、`_scaling`、`_rotation` 做 in-place 的缩放/旋转校正。如果 4DGS 用"每个 Gaussian 存一份逐时刻显式参数"的设计（如 Dynamic 3D Gaussians 那种做法），这里就要同步改写每个时刻的参数，代码会变得很啰嗦、容易出 bug。**这一点强烈建议采用"规范 Gaussian（canonical）+ 轻量形变 MLP" 的设计**，而不是逐时刻显式存参数：
   - 规范 Gaussian（`_xyz/_scaling/_rotation` 等）只有一份，全局 BA 校正代码完全不用改。
   - 形变 MLP 只吃 `(规范 xyz, t)` 输出 `Δxyz/Δscale/Δrotation`，在每次 `render()` 之前算一次、用完即弃，不需要持久化每个时刻的状态，也就不需要在 120-137 行那种全局校正逻辑里做任何特殊处理。

### 4.2 推荐架构：规范 Gaussian + 时间条件形变 MLP

对标 Deformable-3D-Gaussians / 4D-GS 里常见的做法，但按上面两点约束做了本系统专属的调整：

- **模型**：小型 MLP `DeformNet(canonical_xyz, t_encoded) -> (Δxyz, Δrot(四元数增量), Δscale)`，输入位置编码用规范 xyz 的傅里叶特征 + t 的傅里叶特征（频段固定即可，不需要预先知道序列总时长——这点很重要，因为 SLAM 是在线的，事先不知道视频会跑多久）。
  - 不推荐 HexPlane/K-Planes 这类固定网格特征场：这类方法要求预先知道场景包围盒/时间范围，跟本系统"地图边界随 SLAM 增量式跑动态生长"的特性冲突。逐点 MLP 不依赖固定网格，天然适合增量式场景。
- **接入点**：`gaussian/renderer/__init__.py:render()` 的调用处（`initialize_map()` 214行、`map()` 273行、`color_refinement()` 349行）之前，插入一次形变查询：
  ```python
  dxyz, drot, dscale = deform_net(pc.get_xyz, viewpoint.tstamp)
  # 构造一个"临时形变后的 GaussianModel 视图"或直接把 render() 改成接受可选的形变覆盖参数
  render_pkg = render(viewpoint, pc, background, deform=(dxyz, drot, dscale))
  ```
  CUDA 光栅化器完全不用改，因为它只认最终的 `means3D/scales/rotations` 张量。
- **优化器**：仿照 `color_refinement()` 里给位姿残差/曝光单独开 `self.keyframe_optimizers`（343行）的做法，给 `DeformNet` 参数也开一个独立的 `self.deform_optimizer`，在线阶段（`initialize_map`/`map`）和离线阶段（`color_refinement`）都会 `step()`，但学习率/正则强度可以不同——离线阶段可以更激进地拟合全局时间一致性。
- **静态/动态的隐式区分**：不引入显式的"是否动态"标签，而是对 `Δxyz/Δrot/Δscale` 加 L1/L2 正则（鼓励接近零形变，即默认静止），只有光度损失确实需要形变时模型才会让某些 Gaussian 动起来。这是 Deformable-GS 类工作的常见做法，避免一开始就要做运动分割。

### 4.3 一个必须处理的架构性矛盾：跟踪要"去掉动态"，建图要"留住动态"

这是本系统做 4DGS 时**独有**的问题，纯离线 4DGS 论文不会遇到：

- DROID-SLAM 的前端/BA（`motion_filter.py`、`droid_frontend.py`、`geom/ba.py`）估计相机位姿时，隐含假设场景是刚性静止的，动态物体只会被当噪声/outlier 处理（鲁棒核降权，而非显式建模）。如果画面里动态内容占比较大（比如车流密集的路口），仅靠鲁棒核可能不够，位姿会被动态物体"带偏"。
- 但 4DGS 建图恰恰是想**显式重建**这些动态内容，不能把它们当噪声丢掉。
- 也就是说：**同一批像素，跟踪阶段想排除，建图阶段想保留**。需要一个动态/静态分割掩码（可以用现成的语义分割/光流残差/2D 检测器，不需要自己训练），分割结果分别喂给两边：跟踪侧用掩码抑制动态像素对位姿估计的贡献；建图侧用掩码告诉 `DeformNet` 哪些区域"更可能需要形变"（可以只是给正则项加权，不是硬性开关）。
- 这块目前完全没有对应代码（`motion_filter.py`/`droid_frontend.py` 里没有任何 mask 相关的输入通道），是 4DGS 集成里**除了形变模型本身之外，工作量最大、也最容易被低估的部分**。

### 4.4 与全景改造（路线 A，cubemap）的组合关系

两者可以正交叠加，互不冲突：

- cubemap 只影响"渲染哪个虚拟子相机视角"，不影响 Gaussian 本身；`DeformNet` 只吃"规范 Gaussian 位置 + 时间戳"，不关心是被哪个虚拟 pinhole 子相机渲染的。
- 具体来说：同一个物理时刻 `t` 对应的 N 个 cubemap 虚拟子相机，会共享同一次 `DeformNet(canonical_xyz, t)` 查询结果（同一时刻场景只形变一次，只是从不同角度渲染），实现上只要保证"同一 tstamp 的 N 个虚拟视角复用同一份形变后 Gaussian 张量，不重复算 N 次形变"即可（性能优化点，不是正确性问题）。
- 建议顺序：先把 P1（cubemap 全景）跑通，验证渲染管线没问题；4DGS 的 `DeformNet` 作为独立分支开发和调试（可以先在普通 pinhole 单相机数据上验证形变模型本身收敛），最后再把两者接到一起跑全景动态数据。**不建议一开始就在全景 + 动态数据上联调**，两个新维度同时引入会让调试出问题时无法定位是哪一部分的锅。

---

## 5. 分阶段路线图（建议）

| 阶段 | 内容 | 是否需要 CUDA 改动 | 主要风险 |
|---|---|---|---|
| P0 | 找一份全景数据集（或用现有 pinhole 多相机数据模拟全景，做 cubemap 反向验证 pipeline 通不通） | 否 | 数据可得性 |
| P1 | 路线 A：cubemap 分解 + 复用现有多相机 pinhole pipeline，端到端跑通一个全景序列 | 否 | 接缝一致性、DROID 权重在子图上的精度 |
| P2 | 评估是否需要 spherical_ops（若跟踪也想在全帧等距柱状图上做，而不是拆子图跟踪） | 否 | 雅可比正确性、极区退化、DROID 权重迁移 |
| P3（可选，工作量大）| 路线 B：原生 CUDA 球面光栅化器 | 是 | 全新 CUDA 渲染管线开发 |
| P4a | 4DGS：`DeformNet` 独立开发，先在普通 pinhole 单相机数据上验证形变模型能否收敛、离线 `color_refinement` 阶段效果如何 | 否 | 形变场过拟合/欠拟合、正则强度调参 |
| P4b | 4DGS：动态/静态分割掩码接入跟踪侧（`motion_filter.py`/`ba.py` 降权动态像素）+ 建图侧（喂给 `DeformNet` 正则） | 否 | 掩码质量、跟踪精度是否真的因此提升 |
| P4c（最终目标）| 全景 4DGS：P1（cubemap）+ P4a/b 合并联调 | 否 | 调试复杂度（两个新维度叠加） |

---

## 6. 建议的下一步（用于验证可行性，而非承诺工作量）

1. 明确全景数据来源：是真实全景相机采集，还是用现有多相机 rig（如 Waymo 5 相机）拼接近似全景？这决定 P0 怎么做。
2. 明确精度目标：cubemap 接缝的轻微不连续能否接受？如果需要无缝全景渲染（比如面向可视化/游戏引擎导出），可能从一开始就要往路线 B 走。
3. 先做一个不涉及 SLAM 跟踪的最小实验：拿一张全景图 + 已知位姿，只测"能否用现有 Gaussian 渲染管线通过 cubemap 分解正确重建/渲染该全景视角"，把跟踪部分的不确定性（2.2 节风险点）先隔离出去单独评估。
4. 4DGS 侧先独立验证：找一段有轻微动态物体的 pinhole 单相机序列（不涉及全景），只接入 `DeformNet`，看 `color_refinement` 离线阶段能不能把动态物体的重影/鬼影去掉、渲染质量是否优于纯静态 3DGS baseline。这一步不需要动态/静态分割掩码，先看纯隐式正则化能走多远。
5. 动态/静态分割掩码方案选型（4.3 节）：调研现成的语义分割或运动分割模型能否直接接入 `motion_filter.py` 的输入端，评估额外推理开销对 SLAM 实时性的影响。

---

## 7. P4a 原型实验计划：先在 pinhole 单相机数据上验证 DeformNet

目标：**在不碰跟踪、不碰全景、不做动态分割掩码的前提下**，用最小改动验证"规范 Gaussian + 时间条件形变 MLP"这个设计本身能不能收敛、能不能改善动态物体的重建质量。这一步的产出是"go / no-go"判断，不是最终实现。

### 7.1 成功判据（预先定义，避免做完了不知道算不算成功）

- **必要条件**（不满足就说明设计有根本问题，不用往下走）：关掉正则项、在极小的调试子集上做纯过拟合实验，`DeformNet` 能把训练损失降到接近静态 baseline 做不到的水平（证明梯度确实能通过 `render()` 传到 MLP 参数）。
- **核心判据**：在选定的短序列上，全流程跑完后，对比"是否开启 DeformNet"两组结果：
  - 动态物体所在区域裁出来单独算 PSNR/SSIM/LPIPS（不要只看全图平均，全图里动态物体占比小，改善会被稀释掉看不出来）。
  - 目视检查：静态 3DGS baseline 里动态物体处通常会出现"重影/鬼影/糊成一片"（因为同一批 Gaussian 要同时解释物体在多个时刻的多个位置），开启 DeformNet 后应该能看到鬼影明显减少。
- **失败也是有效结果**：如果开了正则项后模型直接把所有 Δ 学成 0（退化成静态 3DGS），说明正则强度或学习率需要调，不代表方向错了；但如果关掉正则项模型依然学不出合理形变（比如损失不降、或输出 NaN/发散），才说明设计或实现有问题，需要回头排查。

### 7.2 数据选择

- 优先直接复用现成、已经跑通的 `data/100613`（本文档 P0 之前我们已经验证过整条 pipeline 能跑）。需要先确认这段序列里有没有明显的动态物体（对向车道车辆、行人等）——如果 47 个关键帧里动态物体占比太小，改善会不明显，不适合做验证。
- 检查方法：不用新写代码，直接看已有的 `output/paper_repro/waymo_100613_jdsa_on/renders` 渲染结果（已经存在于远端），目视找一帧对比 GT 图像，看是否存在因动态物体导致的糊/鬼影区域。如果 100613 动态内容不够明显，换一段别的 Waymo 序列（`paper_experiment_list.md` 里列的几个序列大部分缺原始数据，需要先确认哪几个本地真的有）。
- 不需要专门找"标准动态数据集"（如 Neu3D、HyperNeRF 那种为动态重建设计的数据），因为最终目标场景（自动驾驶多相机 + 全景）本身就是这种"背景静止、少数物体动"的类型，用同类数据验证更贴近实际用例。

### 7.3 代码改动步骤（每步都有独立的可验证检查点，不要一次性全接上再调试）

**Step 1：搭好骨架，但让它在数学上等价于"什么都不做"**
- 新增 `mcgs_slam/gaussian/deform/deform_net.py`：`DeformNet(nn.Module)`，输入规范 `xyz (N,3)` + 标量 `t`，输出 `(Δxyz, Δrot_quat, Δscale)`，内部用傅里叶位置编码 + 3-4 层小 MLP。**最后一层权重/偏置初始化为 0**，保证训练开始前 `Δ≡0`。
- 在 `render()`（`gaussian/renderer/__init__.py:17`）里加一个可选参数，接受形变后的 `means3D/scales/rotations` 覆盖默认值；`gs_backend.py` 里 `initialize_map`/`map`/`color_refinement` 调用 `render()` 前先查一次 `DeformNet`。
- **检查点**：`DeformNet` 参数不参与优化器（先冻结），跑一遍现有 smoke test（`--early_stop 10`），确认渲染结果、PSNR/SSIM/LPIPS、traj 跟"没有 DeformNet 的 baseline"完全一致（数值应该分毫不差，因为 Δ≡0）。这一步只是验证接线正确，不代表模型本身有用。

**Step 2：最小可行的梯度回路验证**
- 解冻 `DeformNet` 参数，加一个独立的 `self.deform_optimizer`（仿照 `color_refinement()` 里 343 行 `self.keyframe_optimizers` 的写法）。
- 不接入真实 pipeline，先写一个几十行的独立调试脚本：固定 2-3 个不同时刻的渲染目标（可以直接用已经跑出来的 GT 图像+已知位姿），只优化 `DeformNet`（Gaussian 本身冻结），关掉所有正则，看损失能不能降下去。
- **检查点**：损失下降曲线正常、没有 NaN/发散。这一步单独脚本跑，几分钟就能出结果，不用等完整 SLAM 流程。

**Step 3：接入在线阶段（`initialize_map`/`map`），先不加正则**
- 在选定序列上跑完整 SLAM 流程（跟踪不变），观察在线阶段（每个关键帧窗口内 `map()` 的 10 次迭代）loss 曲线是否正常、是否有速度明显变慢（`DeformNet` 前向/反向的开销）。
- **检查点**：单关键帧处理时间相比 baseline 增加多少（记录下来，判断是否可接受；论文的完整序列跑一次要几分钟到十几分钟，形变网络前向通常很轻，预期增幅不大）。

**Step 4：接入离线阶段（`color_refinement`），加正则，跑完整对比**
- 打开 `--deform` 开关（新增到 `options.py`），加 `config.yaml` 里 `Training.deform_lr`、`Training.deform_reg_weight`、`Training.deform_pe_freqs` 等超参。
- 用同一段序列，分别跑"开 DeformNet"和"不开（现有 baseline）"两次完整流程（复用我们之前跑通的命令：`--prgbd --jdsa`），对比 7.1 节的判据。
- **检查点**：核心判据是否满足；记录动态物体裁剪区域的 PSNR/SSIM/LPIPS 对比表格，附带渲染图对比截图。

**Step 5（可选，视 Step 4 结果决定要不要做）：超参消融**
- 正则权重（几个数量级）、位置编码频段数、MLP 宽度/深度做小规模网格搜索，找一组能"不牺牲静态区域质量、同时改善动态区域"的配置。

### 7.4 明确不做的事情（避免范围蔓延）

- 不接入动态/静态分割掩码（4.3 节的跟踪侧改动，属于 P4b，等 P4a 验证通过再做）。
- 不跟全景/cubemap 一起联调（4.4 节已说明理由）。
- 不追求任何速度优化，先验证效果，性能优化留到验证通过之后。

### 7.5 工作量量级提示（数量级参考，不是精确承诺）

Step 1-2（骨架+梯度验证）是纯软件工程，风险低；Step 3-4（接入真实 pipeline 出结果）是本原型里最耗时间的部分，因为需要多次完整跑一遍 SLAM+离线精修（参考我们之前的 smoke test，10 帧+30000步离线精修大约 5-6 分钟，完整 47 关键帧序列会显著更长）；Step 5 的消融次数取决于 Step 4 结果好不好，如果 Step 4 一次就达标可以跳过。

---

## 附：本文档依据的关键代码位置索引

| 模块 | 文件 | 关键行 |
|---|---|---|
| 标定加载 | `mcgs_slam/options.py` | 48-68（`load_configs`） |
| 图像流/去畸变 | `mcgs_slam/streams.py` | 21-60 |
| pinhole 投影/反投影 | `mcgs_slam/geom/projective_ops.py` | 19-66 |
| BA/因子图 | `mcgs_slam/geom/ba.py` | 全文件 |
| 关联体 CUDA 核 | `src/correlation_kernels.cu` | 20-70 |
| Gaussian 参数定义 | `mcgs_slam/gaussian/scene/gaussian_model.py` | 32-94 |
| Gaussian 种子化(pinhole反投影) | `mcgs_slam/gaussian/scene/gaussian_model.py` | 113-183 |
| Camera 类 / tstamp 字段 | `mcgs_slam/gaussian/utils/camera_utils.py` | 6-144 |
| 渲染主函数 | `mcgs_slam/gaussian/renderer/__init__.py` | 17-85 |
| CUDA 光栅化器 | `thirdparty/diff-gaussian-rasterization/` | — |
| 在线增量建图（跟踪窗口） | `mcgs_slam/gs_backend.py` | `process_track_data` 72-113；`initialize_map` 211-249；`map` 251-319 |
| 全局位姿/尺度校正（原地改写 Gaussian） | `mcgs_slam/gs_backend.py` | `process_global_track_data` 114-176（尤其120-137） |
| 离线全局精修（4D 一致性优化的合理落点） | `mcgs_slam/gs_backend.py` | `finalize` 178-189；`color_refinement` 321-376 |
