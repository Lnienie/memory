# 基于 LangGraph 的 RAG 知识与经验记忆代理项目

## 一、项目功能介绍

### 1. 项目功能介绍
本项目基于 LangGraph 提供的 ReAct Memory Agent 示例工程进行扩展，目标是构建一个具备长期记忆、知识检索和经验复用能力的智能代理系统。原始项目主要关注用户记忆的保存与召回，即系统会在对话过程中提取用户的偏好、身份信息和长期上下文，并在后续新的会话线程中自动检索这些历史信息，以提升连续对话体验。

在此基础上，本项目进一步引入了更接近 RAG（Retrieval-Augmented Generation，检索增强生成）思想的多源检索机制。系统不再只依赖用户历史对话，而是增加了知识库和经验库两个独立的信息来源。知识库用于保存事实性、说明性和规则性的内容，例如某项技术原理、某个领域背景知识或内部整理的说明文档片段；经验库用于保存可复用的问题处理步骤、排障方法、最佳实践和解决方案，例如常见故障的定位流程或测试问题的排查经验。这样，系统中的信息被划分为用户记忆、知识库和经验库三类，分别服务于个性化上下文、事实知识和问题解决经验。

在运行流程上，系统会在每次模型调用前同时执行三类检索：检索与当前用户相关的历史记忆、检索与当前问题语义相关的知识库内容，以及检索与当前问题相似的经验库内容。随后，系统会将这三类检索结果统一格式化，并注入到模型的提示词上下文中，再交给大模型生成回答。通过这种方式，模型在回答问题时既能够利用长期对话记忆，也能够结合事实知识和历史经验，从而更接近“会记忆、会查资料、会复用经验”的智能体形态。

总体而言，本项目不是单纯的聊天机器人，而是一个具备多源上下文检索能力的记忆增强型代理系统。它保留了原项目的跨线程用户记忆能力，同时扩展了知识检索和经验复用能力，为后续进一步发展为知识助理、排障助理或项目协作助手提供了基础。

### 2. 本项目所参考/基于的论文或项目来源
本项目直接参考并基于以下开源项目与技术路线实现：

1. LangChain AI 官方开源示例项目 `memory-agent`。
   该项目地址为：https://github.com/langchain-ai/memory-agent
   原项目提供了一个基于 LangGraph 的 ReAct 风格记忆代理示例，展示了如何在对话中保存用户长期记忆，并在新线程中进行召回，是本项目的直接基础工程。

2. LangGraph 框架及其状态图式智能体设计思想。
   LangGraph 提供了节点、边、状态和工具路由机制，使得智能体流程可以显式建模。本项目的 `call_model -> store_memory -> call_model` 主流程以及工具路由逻辑均建立在 LangGraph 的图编排机制之上。

3. RAG（Retrieval-Augmented Generation）相关方法论。
   虽然本项目没有完整实现离线文档切分与批量建库流程，但其核心设计思路参考了检索增强生成的基本思想，即先检索，再将相关内容注入模型上下文，以提升回答的事实性、可解释性和复用能力。

4. ReAct 智能体范式。
   原项目采用 ReAct 风格，即模型可在思考和工具调用之间切换。本项目延续了这一机制，并将工具扩展到用户记忆、知识库和经验库的写入场景。

### 3. 本项目在参考论文或项目的基础上做了哪些改动？
相较于原始的 `memory-agent` 项目，本项目做了以下几方面改动：

1. 将原本单一的用户记忆存储扩展为三类独立存储空间，即用户记忆、知识库和经验库。三类信息分别面向个性化上下文、事实知识和问题解决经验，避免所有内容混杂在同一种数据结构中。

2. 在 `tools.py` 中新增了 `upsert_knowledge` 和 `upsert_experience` 两类写入工具，并保留原有的 `upsert_memory`。这样系统不仅能保存“用户说过的话”，还能够显式保存知识片段和经验片段。

3. 在 `graph.py` 中将原来的单源记忆检索改造成多源联合检索。模型调用前不再只查询用户记忆，而是同时查询用户记忆、知识库和经验库，并将三类结果统一格式化后注入提示词。

4. 修改了提示词上下文组织方式。原项目主要围绕用户记忆构造上下文，本项目增加了更通用的 `memory_context` 注入方式，使提示词可以承载多种来源的检索结果。

5. 扩展了工具路由逻辑。原项目仅支持 `upsert_memory` 的执行，本项目新增了对 `upsert_knowledge` 与 `upsert_experience` 的识别和执行，使模型能够根据任务需要选择不同的存储目标。

6. 增加了测试覆盖。测试不再只检查用户记忆是否能保存与召回，还新增了知识库存储、经验库存储、多源检索命中、空库降级以及路由行为验证，从而确保新能力可用且不会破坏旧能力。

需要说明的是，本项目虽然已经具备了知识检索和经验复用能力，但目前的“RAG”更偏向基于 LangGraph Store 的内置检索增强，而不是完整的“上传文档 -> 自动切块 -> 自动建索引 -> 独立知识库管理界面”产品形态。这一部分可作为后续进一步扩展的方向。

## 二、实验结果
本项目完成后，分别从代码质量、类型检查、功能正确性和集成测试几个维度进行了验证。首先，在静态检查方面，项目通过了 `ruff` 的代码规范检查，说明新增代码在导入组织、文档字符串与基本代码风格上符合当前仓库的规范要求。其次，在类型检查方面，项目通过了 `mypy --strict src`，说明扩展后的核心代码在静态类型层面没有明显问题。再次，在功能测试方面，项目通过了 `tests/integration_tests/test_graph.py` 中的 7 项集成测试，覆盖了原有用户记忆功能、知识库存储、经验库存储、多源检索联合命中、空知识库/经验库情况下的降级行为以及路由逻辑的正确性。

从实际运行效果来看，系统在 LangGraph Studio 中可以完成基础对话，并保持原有跨线程用户记忆能力不受影响。在此基础上，当向系统输入适合作为知识片段或经验片段的内容时，模型具备调用对应工具将内容写入不同存储命名空间的能力；在后续新线程提问时，系统能够在生成回答前自动查询与当前问题语义相关的历史信息，并将用户记忆、知识库和经验库中的相关条目一起作为上下文提供给模型。因此，实验结果表明，本项目已经实现了“长期记忆 + 知识检索 + 经验复用”的预期目标，并在不破坏原有基础功能的前提下完成了对原项目的扩展。

## 三、项目运行说明

### 1. 环境准备
本项目运行环境为 Python 项目，主要依赖 LangGraph、LangChain、LangChain OpenAI、LangGraph SDK 等组件。推荐使用 Conda 环境进行隔离管理。本地已使用名为 `memory-agent` 的 Conda 环境运行和测试项目。进入项目目录后，可使用以下命令激活环境：

```powershell
conda.exe shell.powershell hook | Out-String | Invoke-Expression
conda activate memory-agent
```

如果在 Windows 中文环境下运行 `langgraph dev` 时遇到 `UnicodeDecodeError: 'gbk' codec can't decode byte` 之类的编码错误，需要在启动前设置 Python UTF-8 模式：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

### 2. 配置环境变量
项目根目录提供了 `.env.example` 文件，可复制为 `.env` 后填写模型服务所需的密钥。当前项目默认使用 DashScope 兼容 OpenAI 接口的方式调用 Qwen 模型，典型配置如下：

```env
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your-dashscope-api-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=dashscope/qwen-plus
```

其中 `DASHSCOPE_API_KEY` 需要替换为真实可用的密钥。如果使用其他模型服务，也可以根据 LangChain 支持的 provider/model 格式调整 `MODEL` 配置。

### 3. 启动项目
完成环境变量配置后，可在项目根目录执行以下命令启动 LangGraph 本地开发服务：

```powershell
conda.exe shell.powershell hook | Out-String | Invoke-Expression
conda activate memory-agent
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
langgraph dev
```

服务启动后，终端会显示本地访问地址，例如：

```text
http://127.0.0.1:2024
```

随后可以在 LangGraph Studio 中打开当前项目的 graph，并选择 `memory_agent` 进行对话测试。界面中可以观察图结构、线程运行过程、工具调用结果和记忆存储情况。

### 4. 前端交互复现方式
在 LangGraph Studio 页面中，可以通过对话方式测试本项目的三类能力。首先测试用户记忆能力：

```text
我叫张三，我喜欢简洁的回答，平时主要使用 PostgreSQL。请记住这些信息。
```

随后新建一个线程并提问：

```text
你还记得我叫什么吗？我喜欢什么样的回答风格？
```

如果模型能够回答出姓名和偏好，说明用户记忆召回正常。其次测试知识库能力，可输入：

```text
请调用知识库存储工具保存下面内容，不要保存成用户记忆：
标题：PostgreSQL MVCC
内容：PostgreSQL 使用 MVCC 提供事务隔离，读写通常不会互相阻塞。
来源：internal-wiki
```

然后新建线程提问：

```text
PostgreSQL 为什么读写通常不会互相阻塞？请根据知识库回答。
```

若回答中能体现 MVCC、事务隔离、读写不互相阻塞等信息，则说明知识库检索增强生效。最后测试经验库能力，可输入：

```text
请调用经验库存储工具保存下面内容，不要保存成用户记忆：
问题：异步 pytest 卡住
方案：优先检查未 await 的协程和未关闭的后台任务。
上下文：Python 测试排障
```

之后新建线程提问：

```text
我的异步 pytest 一直卡住，应该先排查什么？请优先复用已有经验。
```

若模型能给出检查未 await 协程、后台任务未关闭等建议，则说明经验复用能力正常。

## 四、测试步骤

### 1. 代码规范检查
在项目根目录执行以下命令：

```powershell
conda.exe shell.powershell hook | Out-String | Invoke-Expression
conda activate memory-agent
python -m ruff check .
```

预期结果为：

```text
All checks passed!
```

### 2. 类型检查
执行以下命令进行静态类型检查：

```powershell
conda.exe shell.powershell hook | Out-String | Invoke-Expression
conda activate memory-agent
python -m mypy --strict src
```

预期结果为：

```text
Success: no issues found in 8 source files
```

### 3. 集成测试
执行以下命令运行集成测试：

```powershell
conda.exe shell.powershell hook | Out-String | Invoke-Expression
conda activate memory-agent
python -m pytest tests/integration_tests
```

预期结果为：

```text
7 passed
```

集成测试主要验证以下内容：

1. 原有用户记忆可以正常保存和按 `user_id` 召回。
2. 知识库内容可以写入独立命名空间。
3. 经验库内容可以写入独立命名空间。
4. 当前问题可以同时触发用户记忆、知识库和经验库的联合检索。
5. 当知识库或经验库为空时，系统仍然可以正常使用已有上下文继续运行。
6. 工具路由只处理受支持的记忆、知识和经验写入工具，不会误处理无关工具调用。

### 4. 测试中可能遇到的问题
如果运行测试时提示 `No module named pytest`，说明当前环境未安装测试依赖，可执行：

```powershell
python -m pip install pytest
```

如果运行旧版测试时出现 LangSmith token 无效的问题，说明测试尝试连接 LangSmith 云端服务。当前项目中的测试已改成本地可运行断言，不再依赖有效的 LangSmith API Key。
