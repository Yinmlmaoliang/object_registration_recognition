### Section 3: Methodology

#### 3.1 Method Overview (方法概述)

+ **核心叙述**：提出一个 **"Personalized Object Learning Framework"**。
+ **数学形式化**：定义问题。给定一个自然的手持物体视频序列 $ V $ 和用户提供的自然语言描述 $ T_{desc} $，系统的目标是学习一个多模态模版 $ O = \{f_{vis}, f_{sem}\} $用于物体的个性化表示。
  - $ f_{vis} $: 细粒度的视觉指纹（Visual Fingerprint）。
  - $ f_{sem} $: 个性化的语义属性（Semantic Attributes）。
+ **流程概览图**：文字配合一张流程图，展示三个阶段：
  1. **Acquisition**: 从手持交互中捕获物体。
  2. **Representation**: 构建视觉-语义双通道模版。
  3. **Inference**: 基于 LLM 的意图解析与混合检索。

注：流程图忽略

#### 3.2 Interactive Object Registration

这部分描述底层的感知 Pipeline（RexOmni + SAM2），强调“如何从自然交互中获取干净信号”。

+ **Natural Handheld Interaction**: 描述用户只需自然手持物体，无需严格背景或标注。
+ **Promptable Detection & Segmentation**:
  - 介绍使用 `RexOmni` 响应 "item held by hand" 的 Prompt。
  - 介绍利用 `SAM2` 进行 Temporal Consistency 的分割，从视频流中提取出高质量的物体 Mask，去除人手和背景干扰。
+ Dual-Channel Object Representation (双通道物体表示 —— **核心创新点**)
  - 详细阐述“视觉”与“语义”如何互补。
  - **Channel 1: Fine-Grained Visual Embedding (The "Appearance")**
    * **作用**: 解决 "Is this the same looking object?" 的问题。
    * **实现**: 简述使用 DINOv3。为了保证特征纯净，我们采用了 Foreground Feature Averaging 策略，确保 visual embedding 仅关注物体本身。
  - **Channel 2: Personalized Semantic Embedding (The "Identity")**
    * **作用**: 解决 "Is this the object the user cares about?" 的问题。弥补视觉特征无法理解 "my favorite", "gift from Alice", "bedroom mug" 等个性化概念的缺陷。
    * **实现**: 使用 SentenceTransformer 对用户的口述属性（Attributes）进行编码。
  - **The Bridge (Fusion Mechanism)**:
    * 论述这种 $ O = (v, t) $ 的结构如何允许系统在识别阶段同时处理外观匹配（Appearance Matching）和概念匹配（Concept Matching）。

#### 3.3 Hybrid Retrieval & Recognition Mechanism (混合检索与识别机制)

描述如何利用上述双通道表示进行推理。

+ **Query Parsing**: 用户的 Query 可以是图像（"这是什么？"）或文本（"找我的水杯"）。
+ **Similarity Metric**: 定义混合相似度得分
+ **Open-Vocabulary Localization**: 描述在测试场景中，如何结合 `RexOmni` 的通用检测能力和个性化库的特征匹配，在复杂场景中定位特定物体。

#### 3.4 System Implementation with LLM Agent (基于LLM Agent的系统实现)

这是你要求的“应用层”描述，展示整个 Pipeline 如何被封装成一个可交互的助手。

+ **Conversational Interface**: 介绍使用 **DeepSeek** 作为大脑，负责理解用户的自然语言指令。
+ **Function Calling & Tool Orchestration**:
  - 定义 Agent 可用的工具集（Tools）：
    * `register_object(image_stream, description)`: 触发 3.2 的流程。
    * `recognize_object(query, scene_image)`: 触发 3.3 的流程。
  - 展示一个 **Prompt Template** 示例，说明如何指导 DeepSeek 提取关键参数（如物体名称、属性描述）并调用底层 Python API。
+ **Workflow**: 描述一个典型的交互闭环：用户说话 -> Agent 解析 -> 调用 Pipeline -> 反馈结果。

---

### 写作过程应该注意的核心创新点 (Contributions) 

1. **Dual-Channel Personalized Representation**: 提出了一种融合细粒度视觉特征（Fine-grained Visual Features）和用户定义语义属性（User-defined Semantic Attributes）的物体表示方法。该方法有效弥补了纯视觉模型无法理解用户个性化语义（如 "my favorite"）的缺陷。
2. **Interactive Object Learning Pipeline**: 设计了一套从手持交互视频中自动提取、分割并注册物体的流水线，利用 SAM2 和 DINOv3 实现了低成本、自然的物体知识获取。
3. **LLM-Driven Conversational Agent System**: 集成 DeepSeek LLM 与 Function Calling 机制，构建了一个具备即时学习能力的具身智能助手，实现了从自然语言指令到底层视觉操作的端到端闭环。