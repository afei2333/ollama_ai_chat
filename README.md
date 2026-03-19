# ollama_ai_chat

基于 Ollama 的个人 AI 助手平台，支持 ReAct 模式、多模型切换、工具调用、QQ Bot 集成。

## 功能特性

- **ReAct 智能模式** — 自动 Thought → Action → Observation 循环，最多 10 步自主完成任务
- **多模型支持** — 本地 Ollama 模型（默认 qwen3.5:9b）+ Google Gemini API
- **内置工具** — Shell 命令执行、网页搜索（Tavily）、文件读写、Python 代码运行
- **会话管理** — 多会话、历史记录持久化、Markdown 导出
- **知识库** — 文档存储与关键词检索
- **QQ Bot** — 接入腾讯官方 QQ Bot API，支持多账号、私聊/群聊
- **流式响应** — SSE 实时 token 输出，支持中断停止
- **笔记服务** — 独立 HTTP 服务，可视化任务规划

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端 | Python 3, FastAPI, Uvicorn |
| LLM | Ollama（本地）+ Google Gemini API |
| 数据库 | SQLite (WAL 模式) |
| 前端 | HTML5 + JavaScript + SSE |
| QQ Bot | 腾讯官方 QQ Bot API (WebSocket) |
| 搜索 | Tavily Search API |

## 项目结构

```
ollama_ai_chat/
├── app.py              # FastAPI 入口，API 路由
├── config.py           # 全局配置、工具定义、ReAct 提示词
├── core/               # 核心业务逻辑
│   ├── models.py       # 数据模型（SessionState, KnowledgeDoc）
│   ├── storage.py      # SQLite 持久化层
│   ├── chat.py         # 对话管理，ReAct 循环编排
│   └── logger.py       # 日志配置（轮转日志）
├── llm/                # LLM 后端适配
│   └── google.py       # Google Gemini API 适配器
├── tools/              # 工具执行层
│   ├── executor.py     # 工具执行器（shell、搜索、文件等）
│   └── parser.py       # 工具调用解析工具
├── bot/                # QQ Bot 集成
│   ├── client.py       # 腾讯官方 QQ Bot API 客户端
│   └── runner.py       # QQ Bot 多账号主程序
├── notes/              # 笔记服务
│   └── server.py       # HTTP 服务（端口 8765）
├── skills/             # 可扩展技能模块（股票数据等）
├── static/
│   └── index.html      # 前端 Web UI
├── data/
│   └── assistant.db    # SQLite 数据库
├── workspace/          # Agent 文件操作工作目录
├── logs/               # 应用日志 & LLM 日志
├── requirements.txt
├── .env                # 环境变量配置
└── bots.json           # QQ Bot 多账号配置
```

## 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.ai/) 已安装并运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

复制并编辑 `.env` 文件：

```env
TAVILY_API_KEY=your_tavily_key       # 网页搜索（可选）
OLLAMA_HOST=http://localhost:11434   # Ollama 服务地址
GOOGLE_API_KEY=your_google_key       # Google Gemini（可选）
GOOGLE_HTTP_PROXY=http://...         # Google 代理（可选）
```

### 启动主服务

```bash
uvicorn app:app --reload
```

访问 `http://localhost:8000` 打开 Web UI。

### 启动 QQ Bot（可选）

配置 `bots.json` 后运行：

```bash
python -m bot.runner
```

### 启动笔记服务（可选）

```bash
python -m notes.server
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/models` | 获取可用模型列表 |
| GET | `/config/tools` | 工具可用状态 |
| GET/POST | `/sessions` | 会话列表/创建 |
| PATCH | `/sessions/{id}` | 更新会话配置 |
| DELETE | `/sessions/{id}` | 删除会话 |
| GET | `/sessions/{id}/messages` | 获取历史消息 |
| GET | `/sessions/{id}/export` | 导出 Markdown |
| POST | `/sessions/{id}/stream` | 流式对话（SSE）|
| POST | `/sessions/{id}/abort` | 中断生成 |
| GET/POST | `/kb/docs` | 知识库文档管理 |
| DELETE | `/kb/docs/{id}` | 删除文档 |

## ReAct 模式说明

启用 ReAct 模式后，模型会自动按照以下格式循环执行：

```
Thought: 分析当前问题...
Action: run_shell
Action Input: {"command": "ls -la"}
Observation: [工具执行结果]
... (最多 10 步)
Final Answer: 最终回答
```

SSE 事件类型：`token`、`react_step`、`tool_call`、`error`

## QQ Bot 命令

| 命令 | 说明 |
|------|------|
| `/stop` | 中断当前生成 |
| `/new` | 开始新对话 |

## 配置说明

`config.py` 中的主要常量：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_MODEL` | `qwen3.5:9b` | 默认 Ollama 模型 |
| `DEFAULT_MAX_TURNS` | `8` | 默认上下文轮数 |
| `REACT_MAX_STEPS` | `10` | ReAct 最大步数 |
| `LLM_NUM_CTX` | `8192` | 上下文窗口大小 |
| `LLM_NUM_PREDICT` | `2048` | 最大生成 token 数 |

## 开发计划

### 0306
- [x] 创建聊天界面
- [x] 可以切换模型
- [x] 可以调用命令
- [x] 可以联网搜索
- [ ] 开放系统提示词到前端，方便用户自定义
- [x] 添加删除聊天的按钮

### 0311
- [ ] skills 模块
- [x] 加入对 Gemini 模型的支持
- [x] 接入 QQ Bot

### 0312
- [x] 前端/qqbot 中断生成逻辑 `/stop`，qqbot 发起新对话的逻辑 `/new`

### 0313
- [x] 升级为 ReAct 模式
