# ObjectRegistrar 测试脚本使用说明

## 概述

`test_registrar.py` 是一个完整的测试脚本，用于测试 `ObjectRegistrar` 物体注册功能，包括：

- **One-shot 注册**: 单张图像物体注册
- **N-shot 注册**: 多张图像物体注册
- **调试可视化**: 检测框、分割蒙版的可视化
- **模板保存/加载**: 测试模板的持久化功能

## 运行环境

确保在正确的 conda 环境中运行：

```bash
conda activate rexomni
```

## 运行测试

### 基本用法

```bash
python test_registrar.py
```

或者（如果已设置可执行权限）：

```bash
./test_registrar.py
```

## 测试内容

### Test 1: One-shot 注册

- 使用 `examples/handheld/cellphone1/` 中的第一张图像
- 注册为 `cellphone1_oneshot` 实例
- 生成单个物体特征模板
- 保存可视化结果到 `debug_output/one_shot/`

### Test 2: N-shot 注册

- 使用 `examples/handheld/cellphone1/` 中的所有图像（5张）
- 注册为 `cellphone1_nshot` 实例
- 为同一物体生成多个特征模板
- 保存每张图像的可视化结果到 `debug_output/n_shot/`

### Test 3: 模板保存/加载

- 将注册的模板保存到文件
- 创建新的 Registrar 实例并加载模板
- 验证保存和加载的一致性
- 模板文件保存到 `debug_output/templates/`

## 输出目录结构

运行脚本后，会在项目根目录下创建 `debug_output/` 目录：

```
debug_output/
├── one_shot/                    # One-shot 注册可视化
│   └── cellphone1_oneshot_00000579.jpg
├── n_shot/                      # N-shot 注册可视化
│   ├── cellphone1_nshot_img01_00000579.jpg
│   ├── cellphone1_nshot_img02_00000593.jpg
│   ├── cellphone1_nshot_img03_00000646.jpg
│   ├── cellphone1_nshot_img04_00000671.jpg
│   └── cellphone1_nshot_img05_00000810.jpg
└── templates/                   # 保存的模板文件
    ├── cellphone1_templates.pkl   # 特征嵌入（pickle格式）
    └── cellphone1_metadata.json   # 元数据（JSON格式）
```

## 可视化说明

每张可视化图像包含：

1. **绿色边框**: 检测到的物体边界框
2. **绿色蒙版**: 分割出的物体区域（半透明叠加）
3. **轮廓线**: 物体的精确轮廓
4. **文本标签**:
   - 检测框信息
   - 分割置信度分数
   - 使用的检测提示词

## 模板文件格式

### templates.pkl

Pickle 格式的特征嵌入文件：

```python
{
    'instance_id': np.ndarray,  # Shape: (N_templates, feature_dim)
    ...
}
```

### metadata.json

JSON 格式的元数据文件：

```json
{
  "instance_id": [
    {
      "bbox": [x0, y0, x1, y1],
      "seg_score": 0.95,
      "prompt": "item held by hand"
    },
    ...
  ]
}
```

## 成功标准

脚本会输出每个测试的结果：

- ✓ PASSED: 测试成功
- ✗ FAILED: 测试失败

最终返回码：
- `0`: 所有测试通过
- `1`: 至少一个测试失败

## 典型输出示例

```
================================================================================
OBJECT REGISTRAR TEST SUITE
================================================================================

Output directory: /path/to/debug_output
  - one_shot/: Single image registration visualizations
  - n_shot/: Multiple image registration visualizations
  - templates/: Saved template files

Found 5 handheld images:
  - 00000579.jpg
  - 00000593.jpg
  - 00000646.jpg
  - 00000671.jpg
  - 00000810.jpg

================================================================================
TEST 1: ONE-SHOT REGISTRATION
================================================================================
Image: 00000579.jpg
Instance ID: cellphone1_oneshot
Prompt: 'item held by hand'
--------------------------------------------------------------------------------

Registering object: cellphone1_oneshot
  Prompt: 'item held by hand'
  [Stage 1] Detecting object...
  [Stage 1] Detected 1 object(s)
            Selected bbox: ['123.4', '45.6', '789.0', '234.5']
  [Stage 2] Segmenting object...
  [Stage 2] Segmentation complete (score: 0.963)
  [Stage 3] Extracting features...
  [Stage 3] Feature extracted (dim: 768)
            Norm: 1.0000
  [Stage 4] Storing template...
  [Stage 4] Template stored (1 total)
  [SUCCESS] Registration complete!

--------------------------------------------------------------------------------
Result: RegistrationResult(success=True, instance_id='cellphone1_oneshot', feature_dim=768)
SUCCESS: Object 'cellphone1_oneshot' registered successfully!
  Feature dimension: 768
  Detection bbox: [123.4, 45.6, 789.0, 234.5]
  Segmentation score: 0.9630
  Saved visualization: debug_output/one_shot/cellphone1_oneshot_00000579.jpg
================================================================================
```

## 故障排查

### 错误: No images found

- 检查 `examples/handheld/cellphone1/` 目录是否存在
- 确认目录中有 `.jpg` 图像文件

### 错误: CUDA out of memory

- 减少批处理大小或使用更小的模型
- 或者修改脚本使用 CPU：`device="cpu"`

### 错误: Model loading failed

- 确认模型文件已下载：
  ```bash
  cd models/sam2/checkpoints
  bash download_ckpts.sh
  ```
- 检查 `models/Rex-Omni/` 目录是否存在

## 自定义测试

### 修改测试图像

编辑 `test_registrar.py` 中的路径：

```python
handheld_dir = PROJECT_ROOT / "examples" / "handheld" / "your_object_dir"
```

### 修改检测提示词

在函数调用中修改 `prompt` 参数：

```python
test_one_shot_registration(
    registrar=registrar,
    image_path=image_path,
    instance_id="my_object",
    output_dir=output_dir,
    prompt="your custom prompt"  # 修改这里
)
```

### 调整可视化参数

修改 `save_registration_debug()` 函数中的参数：

```python
img_with_result = draw_detection_with_segmentation(
    image=image,
    pred_bbox=bbox,
    gt_bbox=bbox,
    mask=mask,
    iou=1.0,
    prompt_text=f"Custom label",
    mask_color=(0, 255, 0),  # RGB颜色
    mask_alpha=0.4,          # 透明度 (0.0-1.0)
    show_labels=True
)
```

## 进一步开发

### 扩展到多个物体类别

```python
# 注册多个不同的物体
for object_name in ['cellphone', 'mug', 'bottle']:
    object_dir = handheld_dir / object_name
    image_files = sorted(object_dir.glob("*.jpg"))

    test_n_shot_registration(
        registrar=registrar,
        image_paths=image_files,
        instance_id=object_name,
        output_dir=output_dir
    )
```

### 集成到评估流程

```python
from src.recognition.object_recognizer import ObjectRecognizer

# 1. 注册阶段
registrar.register_batch(support_images, instance_id="object1")
registrar.save_templates("templates/object1")

# 2. 识别阶段
recognizer = ObjectRecognizer(...)
recognizer.load_templates("templates/object1")
results = recognizer.recognize(query_image)
```

## 相关文件

- `src/registration/object_registrar.py`: ObjectRegistrar 实现
- `src/utils/result_analyzer.py`: 可视化工具
- `src/model_loader.py`: 模型加载器
- `CLAUDE.md`: 项目整体说明文档

## 许可证

与主项目相同。
