# MixamoSketch2ARKit

把 **Mixamo 角色模型** 转换成 **RealityKit / ARKit `BodyTrackedEntity` 可驱动的人形 USDZ**。

Swift 端直接这样用：

```swift
let entity = try await Entity(named: "MyCharacter_ARKit", in: bundle)
let body = entity as! BodyTrackedEntity      // 已被 ARKit 人体追踪驱动
let anchor = AnchorEntity(.body)
anchor.addChild(body)
arView.scene.addAnchor(anchor)
```

---

## 1. 为什么不能"只改骨骼名字"

RealityKit **不做重定向**：ARKit 把 `ARSkeletonDefinition.defaultBody3D` 的 91 个关节变换**按名字**套到模型骨架上。所以模型必须真正是 Apple 的骨架：

| 必须一致 | 不一致的后果 |
| --- | --- |
| 91 个关节名（`hips_joint`、`left_arm_joint`…） | 名字对不上 → 关节不动 / 报错 |
| 关节父子层级 | 姿态层级错乱 |
| 关节绑定朝向（bind pose 旋转） | 每个肢体带固定角度的扭曲 |
| **关节方向**（相对父关节的 offset 方向） | 小腿外八 9°，所有腿部姿态都歪 9° |
| 静止姿态是 T-pose、+Y 上、+Z 前 | 人躺倒 / 背对镜头 / 左右反 |
| ≤ 4 个骨骼影响/顶点、无动画、单一骨架 | Body tracking 拒绝加载 |

因此本项目的做法是：**以 Apple 官方 `biped_robot.usdz` 的骨架为目标骨架**（`Biped-Robot/Biped Robot/biped_robot.usdz`，即 ARKit `defaultBody3D` 的 91 关节），把 Mixamo 的网格、权重、贴图"装"到这个骨架上——而不是给 Mixamo 骨骼改名。

**关节方向取 Apple 的、骨长取角色自己的。** 官方模型是机器人（大腿+小腿占身高 51%、手指特别长），直接照抄长度会把人拉变形；而方向必须和官方完全一致，否则追踪姿态整体偏斜。

---

## 2. 目录结构

```
MixamoSketch2ARKit/
├─ run_all.py                 唯一入口：串起全部步骤，可复跑
├─ requirements.txt
├─ models/                    放 Mixamo 导出的 .fbx（不提交仓库）
├─ build/                     每步中间产物（自动生成）
│   ├─ arkit_skel.json        Apple 参考骨架（全局共用，只生成一次）
│   └─ <slug>/                单个角色的 mesh.npz / mesh_meta.json / tex/ / preview_*.png
├─ dist/                      最终产物 <slug>_ARKit.usdz
├─ Biped-Robot/               Apple 官方参考资产（输入，勿删）
└─ tools/
   ├─ config.py               所有路径的唯一来源（可用环境变量覆盖）
   ├─ fbx_binary.py           纯 Python 二进制 FBX 7.x 解析（无 FBX SDK）
   ├─ fbx_scene.py            FBX 场景图：矩阵 / 绑定姿态 / 蒙皮簇
   ├─ arkit_fit.py            关节拟合、T-pose 重拟合、权重kernel、朝向校验
   ├─ dump_robot_skeleton.py  ① 从 Apple 的 usdz 取 91 关节参考骨架
   ├─ extract_fbx.py          ② 解析 FBX：网格 + 权重 + 贴图，并把 91 关节拟合到角色
   ├─ prep_textures.py        ③ 贴图降采样 / 重压缩
   ├─ build_usd.py            ④ 生成 USD（UsdSkel）+ CreateNewARKitUsdzPackage 打包
   ├─ validate_usdz.py        ⑤ 按 Apple checklist 全量校验 + LBS 功能测试
   ├─ render_preview.py       ⑥ 简易软渲染出 T-pose / 测试姿态预览图
   ├─ reweight_torso.py       可选：手工改过网格后重算躯干权重
   └─ selftest.py             无需 Mixamo 模型的端到端自检
```

---

## 3. 安装

```bash
python -m pip install -r requirements.txt
```

- Python 3.9+（在 3.13 上验证）
- `usd-core` 提供编译好的 `pxr`，不需要自己编译 OpenUSD
- 参考资产：`Biped-Robot/Biped Robot/biped_robot.usdz`（Apple "Biped Robot" 示例）。
  若放在别处，用 `ARKIT_REFERENCE_USDZ` 指定；代码会自动搜索项目内的 `biped_robot.usdz`。

---

## 4. 快速开始

```bash
# 1. 把从 Mixamo 下载的 FBX（T-pose、带蒙皮、binary FBX）放进 models/
#    文件名就是产物名：models/My Hero.fbx -> dist/My_Hero_ARKit.usdz

# 2. 全流程
python run_all.py

# 只处理部分角色 / 从某一步续跑 / 首个失败即停
python run_all.py --only "My Hero" Ch16
python run_all.py --from build_usd
python run_all.py --fail-fast

# 输入输出换位置
python run_all.py --models D:/mixamo --dist D:/out
```

产物：`dist/<slug>_ARKit.usdz`；每个角色的日志在 `build/<slug>/pipeline.log`，
汇总在 `build/summary.json`，预览图在 `build/<slug>/preview_*.png`。

单独跑某一步（调试用）：

```bash
python tools/dump_robot_skeleton.py        # -> build/arkit_skel.json
python tools/extract_fbx.py "My Hero"      # -> build/My_Hero/mesh.npz, mesh_meta.json, tex/
python tools/prep_textures.py "My Hero"
python tools/build_usd.py "My Hero"        # -> dist/My_Hero_ARKit.usdz
python tools/validate_usdz.py              # 校验 dist/ 下所有 usdz
python tools/render_preview.py "My Hero"
```

---

## 5. 每一步在做什么

**① dump_robot_skeleton.py** — 打开 Apple 的 `biped_robot.usdz`，导出 91 个关节名、父子表、
`bindTransforms`（世界矩阵）、`restTransforms`，并断言 `bind == rest × parent_bind`。
所有下游测量都对着同一份 JSON，参考骨架只解析一次。

**② extract_fbx.py** — 纯 Python 读二进制 FBX（不依赖 Autodesk SDK）：

- 网格：`Vertices` / `PolygonVertexIndex`（支持 n-gon 扇形三角化）、`LayerElementNormal`、`LayerElementUV`、
  `LayerElementMaterial`，所有 FBX 映射类型（`ByPolygonVertex` / `ByVertice` / `ByControlPoint` / `AllSame`）。
- 贴图：从 `Video` 节点取内嵌图片，按内容哈希去重；识破 Mixamo"同一个文件挂两个 Texture 节点"的写法。
- 蒙皮：`Deformer(Skin)` → `Cluster` → 骨骼，转成每顶点权重。
- 绑定姿态：合并所有 `Pose` 节点（多网格文件每个网格一个），并检查同一骨骼是否给出矛盾矩阵。
- 坐标：按 `GlobalSettings` 的 up/coord 轴转换到 ARKit 空间（Mixamo 恒为 Y-up，实际是单位矩阵），
  单位 cm→m；随后用绑定姿态自检**朝向**：如果角色面向 -Z 且左手在 -X，整体绕 +Y 转 180°（真正的旋转，
  不会破坏绕序/法线）；若朝向与左右矛盾（不可能由旋转修复）则明确报警。
- 关节拟合：`arkit_fit.build_target()` —— 每个 Mixamo 骨段长度照搬，方向吸附到 Apple 的参考方向；
  3 段脊柱重采样成 7 节、1 段脖子重采样成 4 节；手指末节 `_3` 放在网格尖端，`End` 补 12–18 mm；
  趾尖用网格顶点百分比检测。
- T-pose 重拟合：用**同样的旋转**做一次 LBS，把角色"掰直"成 T-pose（骨骼留在肉里，而不是把骨架挪出网格）。
  重拟合前先沿边做两遍权重模糊（否则整段修正会挤在一条边上把网格撕开），并把左右两侧权重镜像统一。
- 权重修复：脊柱/颈部的 12 个关节用沿 hips→head 链的连续核重新分配（原实现有 11 个权重为 0）；
  指节 `_3` 从 `_2` 按比例接管指尖权重；最后压到 **4 影响/顶点** 并归一化。

**③ prep_textures.py** — baseColor → JPEG q92（需要 alpha 时保留 PNG），normal → PNG，长边 ≤ 2048。

**④ build_usd.py** — 生成 `SkelRoot` 根的 USD，单骨架 + 单网格：
`skeleton` 的关节路径写成 `root/hips_joint/spine_1_joint/...`（层级编码在路径里，和 Apple 一致）；
每个关节保留 Apple 的绑定朝向、使用本角色的关节位置；`hips_joint` 移到原点（ARBodyAnchor 的原点就是髋）；
关节位置写 `bindTransforms`，`restTransforms` 由 `bind × inverse(parent_bind)` 反解；
蒙皮写 `skel:skeleton` / `skel:geomBindTransform` / `skel:jointIndices` / `skel:jointWeights`（vertex 插值，elementSize=4）；
每个材质一个 `GeomSubset` + `UsdPreviewSurface`；最后 `UsdUtils.CreateNewARKitUsdzPackage` 打包（自动把贴图塞进 usdz）。

**⑤ validate_usdz.py** — 见下节。

**⑥ render_preview.py** — 直接从打好的 `.usdz` 里读回骨架和网格，做一次 LBS 软渲染：
T-pose 正面/侧面，以及一个弯肘+抬膝的测试姿态。用来快速肉眼确认"没撕、没穿、贴图没歪"。

---

## 6. 校验内容（`validate_usdz.py`）

全部按 Apple *Validating a Model for Motion Capture* + RealityKit 的实际要求：

- 舞台：`upAxis=Y`、`metersPerUnit=1`、`defaultPrim` 是 `SkelRoot`
- 骨架：恰好 1 个 `Skeleton`、91 关节、名字集合与 `defaultBody3D` 完全一致
- 层级：关节路径里的父子关系逐条对照 Apple 表
- 绑定：`hips_joint` 在原点、`spine_1_joint` 与髋重合、`bind == rest × parent_bind`
- 对齐：**每条骨段方向**与参考的偏差 < 3°；位置残差只做参考打印（那是骨长差异）
- 朝向：每个关节的绑定朝向与 Apple 逐元素一致（< 1e-4）
- 骨长：除 Apple 自带的零长关节外没有塌陷；手指 `_3 → End` 在 8–30 mm
- T-pose：手臂水平、肘与肩同高、腿在 X 方向垂直、双脚在髋下方、左右髋对称
- 蒙皮：`elementSize=4`、vertex 插值、≤ 4 影响、权重归一化、索引 < 91、
  所有可形变关节都有权重（面部/末端关节允许为 0）
- 躯干连续性：`spine_1..7`、`neck_1..4` 每个关节 > 200 顶点有权重
- 左右对称：手臂/腿等部位左右总权重比 < 1.8
- 无动画：没有 `SkelAnimation`、全舞台没有 time sample
- **LBS 功能测试**：绑定姿态下顶点位移 ≈ 0；单独旋转前臂/脊柱/大腿/手指/脖子，
  顶点位移在合理范围且**没有撕裂三角面**；指尖确实由 `_3` 驱动；前臂弯曲不影响对侧手/脚/头

`RESULT: ALL CHECKS PASSED` 才是可发布状态，退出码 0。

---

## 7. 配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MIXAMO_MODELS` | `models/` | 输入 FBX 目录 |
| `MIXAMO_BUILD` | `build/` | 中间产物目录 |
| `MIXAMO_DIST` | `dist/` | 输出 usdz 目录 |
| `ARKIT_REFERENCE_USDZ` | 自动搜索 | Apple 的 `biped_robot.usdz` |
| `ARKIT_SKEL_JSON` | `build/arkit_skel.json` | 参考骨架 JSON |
| `MIXAMO_SKIP` | 空 | 逗号分隔，批处理时跳过的 slug |

`run_all.py` 的 `--models / --build / --dist` 就是设置这些路径。

---

## 8. 自检（没有 Mixamo 模型也能跑）

```bash
python tools/selftest.py          # 生成合成模型 -> build_usd -> validate -> render，然后清理
python tools/selftest.py --keep   # 保留 build/zz_selftest 和 dist/zz_selftest_ARKit.usdz
```

用 Apple 的参考骨架伪造一个"Mixamo 风格"的绑定（Mixamo 骨骼命名、每手 3 节手指），
生成一段管子网格并蒙皮，然后跑真实的 ④⑤⑥。任何 USD 结构、层级、蒙皮、打包回归都会在这里暴露，
不需要等下载角色。它也会顺带验证 `dump_robot_skeleton → arkit_skel.json` 的一致性。

---

## 9. 已知限制 / 注意事项

- **网格会被微调**：为了把角色掰成干净 T-pose，顶点会有几毫米到几厘米的位移（日志里打印最大值与边缘拉伸）。
  这是"方向必须对齐 Apple"的代价；与直接用 Mixamo 原姿态相比，追踪质量提升远大于这点形变。
- **会改写左右权重**：Mixamo 经常两侧权重严重不均（同一姿势两臂形变不同），代码默认镜像左手侧权重，
  镜像只作用于镜像误差 < 3 cm 的顶点。
- **手指**：ARKit 的拇指只有 `Start/_1/_2/End`，没有 `_3`；其余手指 `_3` 落在网格指尖，`End` 是补出来的末端。
- **单位/刻度**：本项目输出米制、`metersPerUnit=1`。Apple 自己的 `biped_robot.usdz` 声明的是 `0.01`
  但变换数值其实是米，属于官方资产的小瑕疵，不必模仿。
- **贴图**：长边上限 2048、baseColor 用 JPEG 压缩，会牺牲一点细节换取体积；
  带 alpha 的（头发/睫毛）保持 PNG 并使用 alpha cutoff。
- **仅支持二进制 FBX**（Mixamo 默认导出即二进制）；ASCII FBX 会直接报错提示重导出。
- **面部/眼球关节**没有权重（Mixamo 没有人脸绑定），它们只需要存在且长度非零。

---

## 10. 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| `cannot find Apple's biped_robot.usdz` | 把 `Biped-Robot/` 放回项目根，或设置 `ARKIT_REFERENCE_USDZ` |
| `no models found in .../models` | 把 Mixamo 的 `.fbx` 放进去；文件名决定产物名 |
| `not a binary FBX (ASCII FBX...)` | Mixamo 下载时选 FBX Binary / 在 DCC 里重导出为 binary |
| `the bind pose does not describe a standing skeleton` | FBX 的 `Pose` 节点被改坏，回 Mixamo 重新下载 |
| `bones missing from bind pose: XxxArm` | 模型不是标准 Mixamo 命名（混了自定义骨骼），需要先统一骨骼名 |
| `LEFT/RIGHT MIRRORED` / `faces -Z` 警告 | 角色轴向不符合 ARKit；能修的情况会自动旋转，修不了要回 Mixamo 重新导出 |
| `body-joint position deviation` 警告 | 只是骨长与机器人参考不同，属正常；关注方向偏差是否 < 3° |
| `no stretched triangles` FAIL | 权重或 T-pose 重拟合把网格撕开了，看 `preview_*.png` 与 `pipeline.log` |
| 在 Vision Pro / iPhone 上不跟随 | 先跑 `python tools/validate_usdz.py`，确认 91 关节名与 `SkelRoot` 结构；确认 App 里用的是 `BodyTrackedEntity` |

---

## 11. 参考

- Apple: *Validating a Model for Motion Capture*（`developer.apple.com/documentation/arkit`）
- Apple: `ARSkeletonDefinition.defaultBody3D`（91 关节名与层级）
- Apple 示例资产：Biped Robot（`biped_robot.usdz`，其骨架即本项目目标骨架）
- RealityKit: `BodyTrackedEntity` / `AnchorEntity(.body)`

## License

见 `LICENSE`。Mixamo 角色与 Apple 示例资产各自受其原始授权约束，本仓库不包含这些模型。
