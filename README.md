# 基于 LangGraph 的 RAG 知识与经验记忆代理项目

## 一、项目功能介绍

### 1. 项目功能介绍

本项目基于 LangGraph 的 ReAct Memory Agent 示例拓展，搭建拥有长期记忆、知识检索与经验复用能力的智能代理。原版仅提取留存用户偏好、身份等对话记忆用于跨会话交互；本项目新增知识库、经验库，分别存储事实规则资料与排障方案、实操经验。模型每次应答前同步检索用户记忆、知识库、经验库三类信息，整合后送入提示词生成回复，不再局限单纯对话，可作为记忆增强型多源检索智能代理，适配各类专业助手场景。

### 2. 本项目所参考/基于的论文或项目来源


原项目地址为：<https://github.com/langchain-ai/memory-agent>

## 三、项目运行说明

### 1. 环境准备

本项目运行环境为 Python 项目，主要依赖 LangGraph、LangChain、LangChain OpenAI、LangGraph SDK 等组件。推荐使用 Conda 环境进行隔离管理。本地已使用名为 `memory-agent` 的 Conda 环境运行和测试项目。进入项目目录后，可使用以下命令激活环境：

```powershell
conda activate 你的环境名字
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
conda activate 你的环境名字
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
python -m ruff check .
```

预期结果为：

```text
All checks passed!
```

### 2. 类型检查

执行以下命令进行静态类型检查：

```powershell
python -m mypy --strict src
```

预期结果为：

```text
Success: no issues found in 8 source files
```

### 3. 集成测试

执行以下命令运行集成测试：

```powershell
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
