# 快速入门指南

这是一个简化的指南，帮助你快速开始使用物体注册和识别系统。

## 前置条件

1. 激活 conda 环境：
```bash
conda activate rexomni
```

2. 确保模型文件已下载：
```bash
cd models/sam2/checkpoints
bash download_ckpts.sh
cd ../../..
```

## 快速开始

### 方式一：使用示例工作流脚本

最简单的方式是运行提供的示例脚本：

```bash
./example_workflow.sh
```

这将自动完成注册和识别的完整流程。

### 方式二：手动执行

#### 步骤 1：注册物体

从手持交互图像中注册物体（使用统一的模板库名称）：

```bash
python register_object.py \
  --image_dir examples/handheld/cellphone1/ \
  --instance_id cellphone1 \
  --output templates/my_objects \
  --save_viz
```

**重要**：`--output` 指定的是模板库名称，多个物体实例可以共享同一个模板库，通过 `--instance_id` 区分不同物体。

**预期结果**：
- 模板文件：`templates/my_objects_templates.pkl`（包含 cellphone1 实例）
- 元数据文件：`templates/my_objects_metadata.json`
- 可视化图像：`templates/my_objects_visualizations/`

#### 步骤 2：识别物体

在查询场景中识别已注册的物体：

```bash
python recognize_object.py \
  --image_dir examples/scence/ \
  --templates templates/my_objects \
  --output recognition_results/ \
  --threshold 0.5
```

**预期结果**：
- 可视化结果：`recognition_results/visualizations/`
- 终端输出显示每张图像的匹配结果和置信度

## 常用命令

### 注册命令

```bash
# 单张图像
python register_object.py --image path/to/image.jpg --instance_id object_name

# 多张图像（推荐）
python register_object.py --image_dir path/to/images/ --instance_id object_name

# 保存可视化
python register_object.py --image_dir path/to/images/ --instance_id object_name --save_viz

# 自定义提示词
python register_object.py --image_dir path/to/images/ --instance_id object_name --prompt "custom prompt"
```

### 识别命令

```bash
# 单张图像
python recognize_object.py --image path/to/query.jpg --templates path/to/templates

# 多张图像（推荐）
python recognize_object.py --image_dir path/to/queries/ --templates path/to/templates

# 调整阈值
python recognize_object.py --image_dir path/to/queries/ --templates path/to/templates --threshold 0.6
```

## 增量添加多个物体

**关键概念**：所有物体实例共享同一个模板库文件（通过 `--output` 指定），不同实例通过 `--instance_id` 区分。

```bash
# 注册第一个物体实例到模板库
python register_object.py \
  --image_dir examples/handheld/cellphone1/ \
  --instance_id cellphone1 \
  --output templates/my_objects

# 向同一模板库添加第二个物体实例（增量添加）
python register_object.py \
  --image_dir examples/handheld/mug/ \
  --instance_id mug1 \
  --output templates/my_objects

# 向同一模板库添加第三个物体实例（增量添加）
python register_object.py \
  --image_dir examples/handheld/bottle/ \
  --instance_id bottle1 \
  --output templates/my_objects

# 从模板库中识别所有已注册的物体
python recognize_object.py \
  --image_dir examples/scence/ \
  --templates templates/my_objects
```

**结果**：`templates/my_objects_templates.pkl` 文件中包含 cellphone1、mug1、bottle1 三个实例的模板。

## 查看帮助

```bash
python register_object.py --help
python recognize_object.py --help
```

## 目录结构

运行脚本后会创建以下目录结构：

```
.
├── templates/                          # 模板库目录
│   ├── my_objects_templates.pkl       # 特征嵌入（包含所有实例）
│   ├── my_objects_metadata.json       # 元数据（包含所有实例）
│   └── my_objects_visualizations/     # 注册可视化（可选）
│       ├── cellphone1_001_00000579.jpg
│       ├── mug1_001_00000123.jpg
│       └── bottle1_001_00000456.jpg
│
└── recognition_results/                # 识别结果
    └── visualizations/                 # 识别可视化
        ├── image1_recognition.jpg
        └── image2_recognition.jpg
```

**说明**：
- `my_objects` 是模板库的统一名称
- 一个模板库文件可以包含多个物体实例（cellphone1, mug1, bottle1等）
- 通过 `instance_id` 区分不同的物体

## 下一步

- 详细文档：查看 [SCRIPTS_README.md](SCRIPTS_README.md)
- 匹配机制：查看 [MATCHING_MECHANISM.md](MATCHING_MECHANISM.md) 了解模板匹配的详细原理
- 项目架构：查看 [CLAUDE.md](CLAUDE.md)
- 测试脚本：查看 `tests/test_registrar.py` 和 `tests/test_recognizer.py`
