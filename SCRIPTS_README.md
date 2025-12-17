# 应用脚本使用指南

本文档介绍如何使用 `register_object.py` 和 `recognize_object.py` 两个应用脚本进行物体注册和识别。

## 目录

- [物体注册脚本 (register_object.py)](#物体注册脚本)
- [物体识别脚本 (recognize_object.py)](#物体识别脚本)
- [完整工作流示例](#完整工作流示例)

---

## 物体注册脚本

### 功能

从手持交互场景图像中提取物体特征并保存为模板，支持增量添加不同物体。

### 基本用法

**单张图像注册**

```bash
python register_object.py --image examples/handheld/cellphone1/00000579.jpg --instance_id cellphone1
```

**多张图像注册 (N-shot learning)**

```bash
python register_object.py --image_dir examples/handheld/cellphone1/ --instance_id cellphone1
```

### 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--image` | 二选一 | 单张图像路径 | - |
| `--image_dir` | 二选一 | 图像目录路径 | - |
| `--instance_id` | 是 | 物体实例唯一标识符 | - |
| `--output` | 否 | 模板保存路径（不含扩展名） | `templates/objects` |
| `--prompt` | 否 | 检测提示词 | `item held by hand` |
| `--save_viz` | 否 | 保存可视化结果 | False |
| `--device` | 否 | 运算设备 (cuda/cpu) | `cuda` |

### 高级用法

**自定义检测提示词**

```bash
python register_object.py \
  --image_dir examples/handheld/mug/ \
  --instance_id mug1 \
  --prompt "cup held by hand"
```

**指定输出路径**

```bash
python register_object.py \
  --image examples/handheld/bottle/001.jpg \
  --instance_id bottle1 \
  --output my_templates/bottle1
```

**保存可视化结果**

```bash
python register_object.py \
  --image_dir examples/handheld/cellphone1/ \
  --instance_id cellphone1 \
  --save_viz
```

### 增量添加功能（重要）

**核心概念**：
- `--output` 参数指定的是**模板库名称**（不是物体名称）
- 多个物体实例共享同一个模板库文件
- 通过 `--instance_id` 参数区分不同的物体实例
- 模板库支持增量添加新实例或为现有实例添加更多样本

**示例：建立包含多个物体的模板库**

```bash
# 第一步：创建模板库并注册第一个实例 cellphone1
python register_object.py \
  --image_dir examples/handheld/cellphone1/ \
  --instance_id cellphone1 \
  --output templates/my_objects

# 第二步：向同一模板库添加第二个实例 mug1（增量添加新物体）
python register_object.py \
  --image_dir examples/handheld/mug/ \
  --instance_id mug1 \
  --output templates/my_objects

# 第三步：为 cellphone1 实例添加更多样本（N-shot 扩展）
python register_object.py \
  --image examples/handheld/cellphone1/new_image.jpg \
  --instance_id cellphone1 \
  --output templates/my_objects

# 第四步：向模板库添加第三个实例 bottle1
python register_object.py \
  --image_dir examples/handheld/bottle/ \
  --instance_id bottle1 \
  --output templates/my_objects
```

**结果**：`templates/my_objects_templates.pkl` 文件中包含三个实例的所有模板：
- cellphone1: 6个模板（5个初始 + 1个新增）
- mug1: 3个模板
- bottle1: 4个模板

### 输出文件

```
templates/
└── my_objects_templates.pkl      # 特征嵌入（numpy数组）
└── my_objects_metadata.json      # 元数据（bbox、seg_score等）
└── my_objects_visualizations/    # 可视化结果（如果使用 --save_viz）
    ├── cellphone1_001_00000579.jpg
    ├── cellphone1_002_00000593.jpg
    └── mug1_001_00000123.jpg
```

### 输出示例

```
================================================================================
Object Registration
================================================================================

Instance ID: cellphone1
Images: 5
Prompt: 'item held by hand'
Output: templates/cellphone1

--------------------------------------------------------------------------------
Loading models...
--------------------------------------------------------------------------------
✓ Models loaded successfully

--------------------------------------------------------------------------------
Registering objects...
--------------------------------------------------------------------------------

[1/5] 00000579.jpg
  ✓ Registered successfully
    BBox: ['156.2', '89.3', '423.7', '512.4']
    Seg score: 0.9630

[2/5] 00000593.jpg
  ✓ Registered successfully
    BBox: ['145.8', '102.1', '398.5', '489.2']
    Seg score: 0.9552

...

================================================================================
Registration Summary
================================================================================
Total images processed: 5
Successful: 5 (100.0%)
Failed: 0

Template Statistics:
  - cellphone1: 5 templates
Total instances: 1
Total templates: 5

--------------------------------------------------------------------------------
Saving templates...
--------------------------------------------------------------------------------
✓ Templates: templates/cellphone1_templates.pkl
✓ Metadata: templates/cellphone1_metadata.json
================================================================================
```

---

## 物体识别脚本

### 功能

从查询场景图像中检测并识别已注册的物体，输出匹配结果和可视化。

### 基本用法

**单张图像识别**

```bash
python recognize_object.py --image examples/scence/image1.jpg --templates templates/cellphone1
```

**多张图像批量识别**

```bash
python recognize_object.py --image_dir examples/scence/ --templates templates/cellphone1
```

### 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--image` | 二选一 | 单张查询图像路径 | - |
| `--image_dir` | 二选一 | 查询图像目录路径 | - |
| `--templates` | 是 | 模板文件路径（不含扩展名） | - |
| `--output` | 否 | 输出目录 | `recognition_results` |
| `--prompt` | 否 | 检测提示词 | `objects` |
| `--threshold` | 否 | 置信度阈值 | `0.5` |
| `--device` | 否 | 运算设备 (cuda/cpu) | `cuda` |

### 高级用法

**调整置信度阈值**

```bash
python recognize_object.py \
  --image examples/scence/image1.jpg \
  --templates templates/cellphone1 \
  --threshold 0.6
```

**指定输出目录**

```bash
python recognize_object.py \
  --image_dir examples/scence/ \
  --templates templates/my_objects \
  --output my_recognition_results/
```

**自定义检测提示词**

```bash
python recognize_object.py \
  --image examples/scence/desk_scene.jpg \
  --templates templates/my_objects \
  --prompt "items on desk"
```

### 输出文件

```
recognition_results/
└── visualizations/
    ├── image1_recognition.jpg
    ├── image2_recognition.jpg
    └── image3_recognition.jpg
```

每张可视化图像包含：
- 检测框（lime=高置信度，orange=低置信度）
- 实例 ID 标签
- 置信度分数
- 分割蒙版（半透明叠加）

### 输出示例

```
================================================================================
Object Recognition
================================================================================

Query images: 3
Templates: templates/my_objects
Prompt: 'objects'
Threshold: 0.5
Output: recognition_results

--------------------------------------------------------------------------------
Loading models...
--------------------------------------------------------------------------------
✓ Models loaded successfully

--------------------------------------------------------------------------------
Loading templates...
--------------------------------------------------------------------------------
✓ Loaded 2 instance(s)
  - cellphone1: 5 templates
  - mug1: 3 templates

--------------------------------------------------------------------------------
Recognizing objects...
--------------------------------------------------------------------------------

[1/3] scene_001.jpg
  Detections: 5
  Matches (>=0.5): 2
  Recognized objects:
    1. cellphone1 (confidence: 0.8523)
       Top-3 matches:
         1. cellphone1: 0.8523
         2. cellphone1: 0.8201
         3. mug1: 0.6123
    2. mug1 (confidence: 0.7845)
       Top-3 matches:
         1. mug1: 0.7845
         2. mug1: 0.7623
         3. cellphone1: 0.5234
  ✓ Saved: recognition_results/visualizations/scene_001_recognition.jpg

[2/3] scene_002.jpg
  Detections: 3
  Matches (>=0.5): 1
  Recognized objects:
    1. cellphone1 (confidence: 0.7234)
  ✓ Saved: recognition_results/visualizations/scene_002_recognition.jpg

[3/3] scene_003.jpg
  Detections: 2
  Matches (>=0.5): 0
  ✓ Saved: recognition_results/visualizations/scene_003_recognition.jpg

================================================================================
Recognition Summary
================================================================================
Total images: 3
Total detections: 10
Total matches (>=0.5): 3
Avg detections per image: 3.33
Avg matches per image: 1.00

Results saved to: recognition_results
  - Visualizations: recognition_results/visualizations/
================================================================================
```

---

## 完整工作流示例

以下示例展示从注册到识别的完整流程：

### 场景 1：单个物体识别

```bash
# 步骤 1：创建模板库并注册物体实例
python register_object.py \
  --image_dir examples/handheld/cellphone1/ \
  --instance_id cellphone1 \
  --output templates/my_objects \
  --save_viz

# 步骤 2：从模板库识别物体
python recognize_object.py \
  --image_dir examples/scence/ \
  --templates templates/my_objects \
  --output results/recognition
```

### 场景 2：多个物体识别

```bash
# 步骤 1：注册第一个物体
python register_object.py \
  --image_dir examples/handheld/cellphone1/ \
  --instance_id cellphone1 \
  --output templates/my_objects

# 步骤 2：注册第二个物体（增量添加）
python register_object.py \
  --image_dir examples/handheld/mug/ \
  --instance_id mug1 \
  --output templates/my_objects

# 步骤 3：注册第三个物体（增量添加）
python register_object.py \
  --image_dir examples/handheld/bottle/ \
  --instance_id bottle1 \
  --output templates/my_objects

# 步骤 4：在查询场景中识别所有物体
python recognize_object.py \
  --image_dir examples/scence/ \
  --templates templates/my_objects \
  --output results/multi_object_recognition \
  --threshold 0.6
```

### 场景 3：逐步扩展模板库

```bash
# 第一阶段：创建模板库，初始注册（3张图像）
python register_object.py \
  --image_dir examples/handheld/cellphone1/batch1/ \
  --instance_id cellphone1 \
  --output templates/my_objects

# 第二阶段：为同一实例添加更多样本（2张图像）
python register_object.py \
  --image_dir examples/handheld/cellphone1/batch2/ \
  --instance_id cellphone1 \
  --output templates/my_objects

# 第三阶段：添加新的物体实例
python register_object.py \
  --image_dir examples/handheld/mug/ \
  --instance_id mug1 \
  --output templates/my_objects

# 此时模板库 templates/my_objects 包含：
#   - cellphone1: 5 个模板
#   - mug1: 3 个模板
```

---

## 错误处理

### 常见错误及解决方法

**1. 模型加载失败**

```
✗ Failed to load models: ...
```

- 确保 SAM2 checkpoints 已下载：
  ```bash
  cd models/sam2/checkpoints
  bash download_ckpts.sh
  ```
- 检查 `models/Rex-Omni/` 目录是否存在

**2. CUDA out of memory**

- 使用 CPU 模式：
  ```bash
  python register_object.py --image ... --device cpu
  ```

**3. 模板文件不存在**

```
✗ Failed to load templates: Templates file not found
```

- 检查模板路径是否正确
- 确保已运行注册脚本生成模板

**4. 未检测到物体**

```
✗ Failed: Detection failed: No objects detected
```

- 尝试调整检测提示词 `--prompt`
- 检查图像中是否确实包含目标物体

---

## 性能优化建议

1. **批量处理**：使用 `--image_dir` 比多次调用 `--image` 更高效
2. **GPU 加速**：确保使用 `--device cuda`（默认）
3. **置信度阈值**：根据应用场景调整 `--threshold`，降低阈值可以提高召回率

---

## 与测试脚本的区别

| 特性 | 测试脚本 | 应用脚本 |
|------|---------|---------|
| 目标用户 | 开发者调试 | 最终用户 |
| 接口方式 | 函数调用 | 命令行参数 |
| 路径配置 | 硬编码 | 灵活配置 |
| 输出格式 | 详细日志 | 简洁终端输出 + 可视化 |
| 模板管理 | 覆盖式 | 增量式 |
| 错误处理 | 基础 | 完善 |

---

## 相关文件

- `src/registration/object_registrar.py` - 注册核心逻辑
- `src/recognition/object_recognizer.py` - 识别核心逻辑
- `src/utils/matcher.py` - 模板匹配器实现
- `src/utils/result_analyzer.py` - 可视化工具
- `src/model_loader.py` - 模型加载器
- `MATCHING_MECHANISM.md` - 模板匹配机制详解
- `CLAUDE.md` - 项目架构文档
