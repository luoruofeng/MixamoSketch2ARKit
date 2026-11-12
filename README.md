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

**手是例外之一**：官方机器人的手是一块"桨"，十根手指的方向和人手差 10–40°（`Start` 是掌心内的掌骨关节，`Start→_1` 是整根手指最长的骨头）。把人手吸附到它上面会把手指掰成扇子+爪形，所以手部关节一律保留角色自己的手指方向，整只手作为刚体跟随手腕（每个手指骨段的旋转变换完全相同 ⇒ 手部网格的边长为 0% 变化）。

**腿是另一个例外**：官方机器人站姿是**屈膝 15°**，而且它"脚踝→前脚掌"这条线比人脚平 12°（两侧脚掌都平贴地面，只是 Mixamo 把 `ToeBase` 放得更靠近鞋底，这条线不能当作"脚掌平面"来用）。把人的腿吸附到机器人上会造成两个肉眼可见的后果，而且会被 T-pose 重拟合**烘进网格**、每一帧追踪都带着：

| 现象 | 原因 |
| --- | --- |
| 整只脚上翘、脚后跟往下沉、鞋底离开地面 | 脚踝→前脚掌的方向被拧到机器人的 −22.85°（实测每个角色自己都是 −26°…−38°），整只脚被抬高 3°–15° |
| 膝盖比正常多弯 8°–15° | 小腿被拧到机器人后倾 9° 的站姿 |

所以腿链只纠正**侧向**分量（外八 / 脚尖外撇），俯仰（膝弯多少、脚掌怎么落地）保留角色自己的——见 `arkit_fit.in_plane()` 与 `DEFAULT_SAGITTAL`。这里的旋转是绕 +Y 的**偏航旋转**而非把 X 分量清零：后者会缩短水平投影、连带改变俯仰，脚尖外撇 20° 的脚会因此再翘 2.7°。

> **要查"某个 Mixamo 骨骼对应哪个 ARKit 关节"**：见 [6 节](#6-生成的-usdz-里到底有什么命名--关节映射)，那里有完整的
> 名字规则、usdz 内部 prim 名、91 个关节名，以及每个骨骼的权重目标 / 关节位置 / 重拟合旋转三层映射表。

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
  3 段脊柱重采样成 7 节、1 段脖子重采样成 4 节；趾尖用网格顶点百分比检测
  （**取所有网格里伸得最远的那一个**：皮肤/裤子在 `ToeBase` 后面的那一簇会把趾尖算到脚后跟去，导致趾骨反向）。
  **腿链**（大腿/小腿/脚/脚趾）只纠正侧向外八，俯仰保留角色自己的：日志会打印
  `stance left knee bend 178.6 (own 178.6) | foot pitch -29.23 (own -29.23)`，
  两个数相等就说明膝弯和脚掌落地角度完全没有被改动。
  **手指**：Apple 的每根手指是 `Start`（掌骨，在掌心）`→ _1`（掌指关节）`→ _2`（近端指间）`→ _3`（远端指间）`→ End`（指尖），
  Mixamo 的 `HandIndex1..3` 正好是后三个指骨、`HandIndex4` 是几乎无权重的指尖，所以 1..3 映射到 `_1/_2/_3`、
  网格指尖是 `End`，掌骨 `Start` 按参考比例（手腕→掌指关节的 29.5%）在掌心合成；手部整条链保留角色自己的方向。
  逐骨对应关系见 **6.4 完整映射表**（权重目标、关节位置来源、重拟合旋转三列）。
- 关节旋转按**链累积**：`D[child] = D[parent] · rot_between(u, D[parent]ᵀ·target)`，保证
  `D[j]·u = target` 且相邻骨段只在关节处弯折。旧的"每个关节各自做最小旋转"不满足组合性，
  两根相邻骨段会得到无关的旋转、把中间的肉撕开（手指就是这么被撕成爪形的）。
- T-pose 重拟合：用**同样的旋转**做一次 LBS，把角色"掰直"成 T-pose（骨骼留在肉里，而不是把骨架挪出网格）。
  重拟合前先沿边做两遍权重模糊（否则整段修正会挤在一条边上把网格撕开），并把左右两侧权重镜像统一；
  导出时某根骨头只在一侧有蒙皮簇（常见于整根手指漏绑）就补一列空权重，让镜像步骤把它填上。
- 权重修复：hips→head 链上的全部 13 个关节（1 髋 + 7 脊柱 + 4 颈 + 头）用沿该链弧长的连续核重新分配
  （旧的"一骨对一关节"实现让 `spine_1/3/5/6`、`neck_2/3/4` 的权重恒为 0）；
  掌骨 `Start` 从手掌权重里按"手腕→掌指关节"斜坡接管掌心远端；最后压到 **4 影响/顶点** 并归一化。

**③ prep_textures.py** — baseColor → JPEG q92（需要 alpha 时保留 PNG），normal → PNG，长边 ≤ 2048。

**④ build_usd.py** — 生成 `SkelRoot` 根的 USD，单骨架 + 单网格：
`skeleton` 的关节路径写成 `root/hips_joint/spine_1_joint/...`（层级编码在路径里，和 Apple 一致）；
每个关节保留 Apple 的绑定朝向、使用本角色的关节位置；`hips_joint` 移到原点（ARBodyAnchor 的原点就是髋）；
关节位置写 `bindTransforms`，`restTransforms` 由 `bind × inverse(parent_bind)` 反解；
蒙皮写 `skel:skeleton` / `skel:geomBindTransform` / `skel:jointIndices` / `skel:jointWeights`（vertex 插值，elementSize=4）；
每个材质一个 `GeomSubset` + `UsdPreviewSurface`；最后 `UsdUtils.CreateNewARKitUsdzPackage` 打包（自动把贴图塞进 usdz）。
每个 prim 的确切名字（`SkelRoot` / `<slug>_skeleton` / `<slug>_mesh` / `Looks` / subset / 贴图文件）见 **6.1 命名规则**。

**⑤ validate_usdz.py** — 见 7 节。

**⑥ render_preview.py** — 直接从打好的 `.usdz` 里读回骨架和网格，做一次 LBS 软渲染：
T-pose 正面/侧面，以及一个弯肘+抬膝的测试姿态。用来快速肉眼确认"没撕、没穿、贴图没歪"。

---

## 6. 生成的 usdz 里到底有什么（命名 + 关节映射）

### 6.1 命名规则

```
models/Whiteclown N Hallin.fbx     --config.slug_of-->  slug = Whiteclown_N_Hallin
build/Whiteclown_N_Hallin/                             mesh.npz / mesh_meta.json / tex/ / preview_*.png
build/Whiteclown_N_Hallin/Whiteclown_N_Hallin_ARKit.usdc   USD stage（打包前）
dist/Whiteclown_N_Hallin_ARKit.usdz                        ← 唯一对外产物
```

`slug` 只保留 `[0-9A-Za-z_-]`，空格 / `.` / 中文等一律变成 `_`，首尾 `_` 去掉。**输入 FBX 的文件名就是产物名**，全流程不再重命名。

usdz 内部 prim 树（`<slug>` 与文件名完全一致，方便按名字取节点）：

| prim 路径 | 类型 | 关键属性 |
| --- | --- | --- |
| `/<slug>` | `SkelRoot` | **defaultPrim**、`kind=component`、`assetInfo.name=<slug>` |
| `/<slug>/<slug>_skeleton` | `Xform` | 骨架父节点，与 Apple 的 `SkelRoot/xform/skeleton` 同构 |
| `/<slug>/<slug>_skeleton/root` | `Skeleton` | `joints` / `bindTransforms` / `restTransforms` / `extent` |
| `/<slug>/<slug>_geom` | `Xform` | 网格父节点 |
| `/<slug>/<slug>_geom/<slug>_mesh` | `Mesh` | 单网格；`primvars:st`、`primvars:normals` 为 `faceVarying`；`skel:skeleton` 指向上面的 Skeleton，`skel:geomBindTransform` = 单位阵 |
| `/<slug>/<slug>_geom/<slug>_mesh/<材质名>` | `GeomSubset` | 一个材质一个；`familyName=materialBind`、`elementType=face`、`displayName` 保留原始材质名 |
| `/<slug>/Looks/<材质名>` | `Material` | `UsdPreviewSurface`；子节点固定为 `Surface` / `uvReader` / `baseColorTex` / `normalTex` |

- **名字清洗**（`build_usd.ident`）：非 `[0-9A-Za-z_]` → `_`，数字开头补 `m_`，撞名追加 `_2`、`_3`。`GeomSubset` 与 `Material` 同名，所以 `Looks/` 里的材质和网格里的 subset 一一对应。
- **贴图命名**：`tex/<材质名>_baseColor.jpg`（材质带 alpha 时改用 `.png` + `opacityThreshold=0.35`）、`tex/<材质名>_normal.png`；同一张图被多个材质引用时按内容 md5 去重，第二份加 `_2` 后缀。
- **关节顺序**：`joints` 数组的顺序 = Apple `biped_robot.usdz` 里的顺序（`root`, `hips_joint`, `left_upLeg_joint`, … `left_handThumbEnd_joint`），`skel:jointIndices` 直接索引这个顺序。顺序只由 `build/arkit_skel.json` 决定，不要手改。
- **关节路径**：父子层级编码在路径里（`root/hips_joint/spine_1_joint/…`），最后一段就是扁平关节名。这既是 Apple 的写法，也是 RealityKit 反推父子关系的唯一依据。

### 6.2 91 个关节名

下表中 `side` 取 `left` / `right`：

| 组 | 关节名 | 个数 |
| --- | --- | --- |
| 根 / 髋 | `root`、`hips_joint`、`spine_1_joint`（与 hips 重合） | 3 |
| 脊柱 | `spine_2_joint` … `spine_7_joint` | 6 |
| 颈 / 头 | `neck_1_joint` … `neck_4_joint`、`head_joint` | 5 |
| 脸 | `jaw_joint`、`chin_joint`、`nose_joint`、`<side>_eye_joint`、`<side>_eyeUpperLid_joint`、`<side>_eyeLowerLid_joint`、`<side>_eyeball_joint` | 11 |
| 手臂 | `<side>_shoulder_1_joint`、`<side>_arm_joint`、`<side>_forearm_joint`、`<side>_hand_joint` | 2×4 |
| 手指 | `<side>_hand<A>Start_joint`、`_1`、`_2`、`_3`、`End`，`A ∈ {Index, Mid, Ring, Pinky}` | 2×20 |
| 拇指 | `<side>_handThumbStart_joint`、`_1`、`_2`、`End`（**没有 `_3`**） | 2×4 |
| 腿 | `<side>_upLeg_joint`、`<side>_leg_joint`、`<side>_foot_joint`、`<side>_toes_joint`、`<side>_toesEnd_joint` | 2×5 |

合计 3+6+5+11+8+40+8+10 = 91。名字集合必须与 `ARSkeletonDefinition.defaultBody3D` 完全一致——RealityKit 不做重定向，只按名字套变换。

### 6.3 三层映射：一个 Mixamo 骨骼对三个 ARKit 角色

| 层 | 代码 | 决定什么 | 映射错的表现 |
| --- | --- | --- | --- |
| ① 权重 | `arkit_fit.weight_joint_map()` | 这块肉跟着哪个关节动 | 关节不动 / 肉被相邻骨带走（手指缩进掌心） |
| ② 关节位置与朝向 | `arkit_fit.build_target()` | 关节摆在哪、相对父关节指向哪 | 每帧追踪姿态整体偏斜（腿外八 9°、脚尖翘起） |
| ③ 重拟合旋转 | `arkit_fit._seg_tail()` → `bone_transforms()` | 掰成 T-pose 时该骨段的网格区域用哪个关节的累积旋转 | 网格被缩短/撕开，或关节跑出肉外 |

- ② 用链累积旋转 `D[child] = D[parent] · rot_between(u, D[parent]ᵀ·target)`，保证 `D[j]·u = target` 且相邻骨段只在关节处弯折。
- ③ 取该骨段**远端**关节的 `D`：一段骨头（`HandIndex1`：掌指关节 → 近端指间）的网格区域必须用自己的远端关节（`handIndex_2_joint`）的旋转。换成近端会套上前一段的旋转——骨盆→髋那 19° 的差会把两条大腿甩成罗圈腿。
- ①②③ 的键是同一套 Mixamo 骨骼名，改动必须一起改；`validate_usdz.py` 与 `selftest.py` 会把这些约束当回归闸门。

### 6.4 完整映射表

躯干与四肢（`L/R` 表示 Mixamo 的 `Left` / `Right`，对应 ARKit 的 `left_` / `right_`）：

| Mixamo 骨骼 | ① 权重落到的关节 | ③ 网格重拟合用的旋转 | 说明 |
| --- | --- | --- | --- |
| `Hips` | `hips_joint` | `spine_2_joint` | 骨盆区域按 hips→spine_2 这一段转 |
| `Spine` | `spine_2_joint` | `spine_3_joint` | 关节位置 = 角色自己的 `Spine` 骨（`SPINE_SEQ` 的 1.00） |
| `Spine1` | `spine_4_joint` | `spine_5_joint` | 位置 = 角色自己的 `Spine1` 骨 |
| `Spine2` | `spine_7_joint` | `neck_1_joint` | 位置 = 角色自己的 `Spine2` 骨；锁骨挂在 `spine_7_joint` |
| `Neck` | `neck_1_joint` | `neck_2_joint` | 位置 = 角色自己的 `Neck` 骨，往上按 `NECK_SEQ` 再拆 4 段 |
| `Head` | `head_joint` | `head_joint` | 叶子关节：位置即旋转 |
| `LeftEye` / `RightEye` | `left_eyeball_joint` / `right_eyeball_joint` | — | 关节放在角色自己的眼球位置，眼球顶点不动 |
| `L/RShoulder` | `<side>_shoulder_1_joint` | `<side>_arm_joint` | |
| `L/RArm` | `<side>_arm_joint` | `<side>_forearm_joint` | 方向吸附参考（1.0） |
| `L/RForeArm` | `<side>_forearm_joint` | `<side>_hand_joint` | 同上 |
| `L/RHand` | `<side>_hand_joint` | `<side>_handIndex_1_joint` | 手掌跟随食指掌指关节 |
| `L/RUpLeg` | `<side>_upLeg_joint` | `<side>_leg_joint` | hips→hip 是横向骨，方向严格照参考 |
| `L/RLeg` | `<side>_leg_joint` | `<side>_foot_joint` | 只纠正侧向外八，俯仰保留角色自己的 |
| `L/RFoot` | `<side>_foot_joint` | `<side>_toes_joint` | 同上 |
| `L/RToeBase` | `<side>_toes_joint` | `<side>_toesEnd_joint` | 同上 |
| `L/RToe_End` | `<side>_toesEnd_joint` | `<side>_toesEnd_joint` | Mixamo 的末端标记，通常无权重 |

四根手指（`F` 为 Mixamo 的 `HandIndex` / `HandMiddle` / `HandRing` / `HandPinky`，`A` 为 ARKit 的 `Index` / `Mid` / `Ring` / `Pinky`，两侧各一份）：

| Mixamo 骨骼 | ① 权重落到的关节 | ③ 网格重拟合用的旋转 |
| --- | --- | --- |
| `L/R<F>1`（掌指关节 MCP） | `<side>_hand<A>_1_joint` | `<side>_hand<A>_2_joint` |
| `L/R<F>2`（近端指间 PIP） | `<side>_hand<A>_2_joint` | `<side>_hand<A>_3_joint` |
| `L/R<F>3`（远端指间 DIP） | `<side>_hand<A>_3_joint` | `<side>_hand<A>End_joint` |
| `L/R<F>4`（几乎无权重的指尖帽） | `<side>_hand<A>End_joint` | `<side>_hand<A>End_joint` |

拇指（Mixamo 的 3 节正好是掌骨/MCP/IP，ARKit 的拇指没有 `_3`）：

| Mixamo 骨骼 | ① 权重落到的关节 | ③ 网格重拟合用的旋转 |
| --- | --- | --- |
| `L/RHandThumb1`（掌骨） | `<side>_handThumbStart_joint` | `<side>_handThumb_1_joint` |
| `L/RHandThumb2`（MCP） | `<side>_handThumb_1_joint` | `<side>_handThumb_2_joint` |
| `L/RHandThumb3`（IP） | `<side>_handThumb_2_joint` | `<side>_handThumb_2_joint` |
| `L/RHandThumb4`（指尖帽） | `<side>_handThumbEnd_joint` | `<side>_handThumb_2_joint` |

只在 `build_target()` 里使用的虚拟键（不进 FBX，位置由网格算出来）：`<side>_tip_<A>`（手指尖，99.5 分位最远顶点，多份网格取伸得最远的那个）、`<side>_tip_Thumb`、`<side>_tip_Toe`（沿脚掌水平方向）。前两者落到 `hand<A>End_joint` / `handThumbEnd_joint`，后者落到 `toesEnd_joint`。

### 6.5 Apple 有、Mixamo 没有的关节

| 关节 | 位置来自 | 权重来自 |
| --- | --- | --- |
| `spine_1_joint` | 与 `hips_joint` 重合（Apple 就是这样） | `torso_kernel()` |
| `spine_3_joint` / `spine_5_joint` / `spine_6_joint` | Mixamo 骨段上的重采样点（`SPINE_SEQ`） | `torso_kernel()`（沿 hips→head 弧长的连续核） |
| `neck_2_joint` / `neck_3_joint` / `neck_4_joint` | `NECK_SEQ`（颈+头的 19% / 45% / 70%） | 同上 |
| `hand<A>Start_joint`（4 根掌骨） | 手腕→掌指关节的 29.5% 处（`START_FRAC`），合成在掌心内 | `palm_starts()`：从 `hand_joint` 的掌心远端按距掌指关节的距离接管 |
| `hand<A>End_joint` / `handThumbEnd_joint` | 网格指尖 | Mixamo 的 `<F>4` / `HandThumb4` |
| `jaw_joint` / `chin_joint` / `nose_joint` / `<side>_eye_joint` / `<side>_eye*Lid_joint` | `head_joint` 的参考偏移 × `height_scale`，`D` 取 `head_joint` | 无（零权重，只需存在且长度非零） |
| `<side>_eyeball_joint` | 角色自己的眼球位置 | `LeftEye` / `RightEye` 蒙皮簇 |

`weight_joint_map()` 里没有对应项的 Mixamo 骨骼（例如 `*Finger*_End` 之类的自定义骨）不会被静默丢掉：日志会打印 `bones without ARKit target: …` 并计入 `mesh_meta.json` 的 `warnings`。

### 6.6 关节位置 / 朝向的两条例外

| 链 | 处理 | 为什么 |
| --- | --- | --- |
| 手（`alpha=0`） | 位置取角色自己的手，方向完全不吸附参考 | Apple 的手是机器人"桨"，手指方向与人手差 10–40°；吸附会把手指掰成扇子+爪形 |
| 腿链（`DEFAULT_SAGITTAL`） | 只把骨段绕 +Y 转进矢状面（`in_plane()`），俯仰 / 膝弯 / 脚掌角全保留 | 机器人站姿屈膝 15°、脚踝→前脚掌比人脚平 12°；吸附会抬高整只脚、每帧多弯 8–15° 的膝 |

想回到"完全吸附参考"，把 `arkit_fit.DEFAULT_SAGITTAL` 改成 `{"leg": False, "foot": False, "toe": False}`；`validate_usdz.py` 会读 `mesh_meta.json` 里的 `fit.stance_sagittal` 自动放宽对应校验项。

---

## 7. 校验内容（`validate_usdz.py`）

全部按 Apple *Validating a Model for Motion Capture* + RealityKit 的实际要求：

- 舞台：`upAxis=Y`、`metersPerUnit=1`、`defaultPrim` 是 `SkelRoot`
- 骨架：恰好 1 个 `Skeleton`、91 关节、名字集合与 `defaultBody3D` 完全一致
- 层级：关节路径里的父子关系逐条对照 Apple 表
- 绑定：`hips_joint` 在原点、`spine_1_joint` 与髋重合、`bind == rest × parent_bind`
- 对齐：**每条躯干/手臂骨段方向**与参考的偏差 < 3°（手部、腿链刻意不吸附，只打印偏差）；
  位置残差只做参考打印（那是骨长差异）
- 脚掌：`left/right_foot_joint → toes_joint`（脚踝→前脚掌）俯仰在 −50°…−8° 之间，
  且与 `mesh_meta.json` 里记录的**角色自己的**俯仰差 < 1.5°、膝弯差在 −3°…+6° 之内
  —— 这两条就是"整只脚上翘 / 膝盖多弯"的回归闸门
- 朝向：每个关节的绑定朝向与 Apple 逐元素一致（< 1e-4）
- 骨长：除 Apple 自带的零长关节外没有塌陷；手指 `_3 → End` 是真实末节指骨
  （中节的 0.4–2.2 倍，参考为 1.0）
- 手指拓扑：`Start→_1` 是整根手指最长的骨段（掌骨），且 `Start/_1/_2/_3/End`
  沿手指轴依次排列、`Start` 位于掌心内——映射错一节会在这里立刻暴露
- T-pose：手臂水平、肘与肩同高、腿在 X 方向垂直、双脚在髋下方、左右髋对称
- 蒙皮：`elementSize=4`、vertex 插值、≤ 4 影响、权重归一化、索引 < 91、
  所有可形变关节都有权重（面部/末端关节允许为 0）
- 躯干连续性：`spine_1..7`、`neck_1..4` 每个关节 > 顶点数的 0.8%（约 200 顶点/2 万顶点）
- 左右对称：手臂/腿等部位左右总权重比 < 1.8
- 无动画：没有 `SkelAnimation`、全舞台没有 time sample
- **LBS 功能测试**：绑定姿态下顶点位移 ≈ 0；单独旋转前臂/脊柱/大腿/手指/脖子，
  顶点位移在合理范围且**没有撕裂三角面**；指尖确实由 `_3` 驱动；前臂弯曲不影响对侧手/脚/头

`RESULT: ALL CHECKS PASSED` 才是可发布状态，退出码 0。

---

## 8. 配置（环境变量）

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

## 9. 自检（没有 Mixamo 模型也能跑）

```bash
python tools/selftest.py          # 生成合成模型 -> build_usd -> validate -> render，然后清理
python tools/selftest.py --keep   # 保留 build/zz_selftest 和 dist/zz_selftest_ARKit.usdz
```

用 Apple 的参考骨架伪造一个"Mixamo 风格"的绑定（Mixamo 骨骼命名、每手 3 节手指），
生成一段管子网格并蒙皮，然后跑真实的 ④⑤⑥。任何 USD 结构、层级、蒙皮、打包回归都会在这里暴露，
不需要等下载角色。它也会顺带验证 `dump_robot_skeleton → arkit_skel.json` 的一致性。

---

## 10. 已知限制 / 注意事项

- **网格会被微调**：为了把角色掰成干净 T-pose，身体顶点会有几毫米到几厘米的位移（日志里打印最大值与边缘拉伸）。
  这是"方向必须对齐 Apple"的代价；与直接用 Mixamo 原姿态相比，追踪质量提升远大于这点形变。
  **手部不做任何重拟合**：日志里的 `hand-region edge stretch` 必须是 `x1.00`。
  **腿链只做侧向校正**：贴图/骨骼不动，网格只被绕身体前后轴转掉外八（脚尖外撇 20° 的角色脚部顶点最多挪十几厘米，
  但这是"站正"本身的位移，不是形变；`visible edge stretch` 仍然在 1.x）。
- **腿链与 Apple 参考的方向偏差（最高约 15°）是刻意的**：机器人参考的站姿不是人的站姿（见 1 节）。如果更在意
  "位置严格贴合参考"而不是"脚掌落地/膝盖自然"，把 `arkit_fit.DEFAULT_SAGITTAL` 改成
  `{"leg": False, "foot": False, "toe": False}`（或只关 `leg`）即可回到完全吸附；校验脚本会自动跟着放宽对应项。
- **会改写左右权重**：Mixamo 经常两侧权重严重不均（同一姿势两臂形变不同），代码默认镜像左手侧权重，
  镜像只作用于镜像误差 < 3 cm 的顶点；若某根骨头只在一侧有蒙皮簇（整根手指漏绑），
  会补一列空权重再按放宽的邻域把它填上。
- **手指**：拇指是 `Start(掌骨)/_1/_2/End`（ARKit 的拇指没有 `_3`）；其余四指
  `_1/_2/_3/End` 对应 Mixamo 的 `HandIndex1/2/3` 和网格指尖，`Start` 是在掌心合成的掌骨关节。
- **单位/刻度**：本项目输出米制、`metersPerUnit=1`。Apple 自己的 `biped_robot.usdz` 声明的是 `0.01`
  但变换数值其实是米，属于官方资产的小瑕疵，不必模仿。
- **贴图**：长边上限 2048、baseColor 用 JPEG 压缩，会牺牲一点细节换取体积；
  带 alpha 的（头发/睫毛）保持 PNG 并使用 alpha cutoff。
- **仅支持二进制 FBX**（Mixamo 默认导出即二进制）；ASCII FBX 会直接报错提示重导出。
- **面部**：Mixamo 只绑了眼球（`LeftEye/RightEye` → `*_eyeball_joint`，关节放在角色自己的眼球位置），
  其余面部关节没有权重，只需要存在且长度非零。

---

## 11. 故障排查

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
| 日志出现 `hand mesh is distorted` | 手指关节没有跟着角色自己的手（映射错位或又把手指吸附到参考方向了），手上不应该有任何重拟合 |
| 手在 AR 里"缩进手掌"/手指粘在一起 | 手指映射错了一节：`*Start` 是掌骨（掌心内），Mixamo 的 `HandIndex1` 是掌指关节 |
| 在 Vision Pro / iPhone 上不跟随 | 先跑 `python tools/validate_usdz.py`，确认 91 关节名与 `SkelRoot` 结构；确认 App 里用的是 `BodyTrackedEntity` |

---

## 12. 参考

- Apple: *Validating a Model for Motion Capture*（`developer.apple.com/documentation/arkit`）
- Apple: `ARSkeletonDefinition.defaultBody3D`（91 关节名与层级）
- Apple 示例资产：Biped Robot（`biped_robot.usdz`，其骨架即本项目目标骨架）
- RealityKit: `BodyTrackedEntity` / `AnchorEntity(.body)`

## License

见 `LICENSE`。Mixamo 角色与 Apple 示例资产各自受其原始授权约束，本仓库不包含这些模型。
