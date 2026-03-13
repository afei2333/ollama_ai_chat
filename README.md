# ollama_ai_chat
build for person ai assistant with ollama

run cmd
```powershell
uvicorn app:app --reload
```

## 开发计划
### 0306
- [x] 创建聊天界面
- [x] 可以切换模型
- [x] 可以调用命令
- [x] 可以联网搜索
- [] 开放系统提示词到前端，方便用户自定义
- [x] 添加删除聊天的按钮
### 0311
- [] skills 模块
- [x] 加入对gemini模型的支持
- [x] 接入QQbot

### 0312
- [x] 前端/qqbot中断生成逻辑/stop，qqbot发起新对话的逻辑/new

### 0313
- [x] 升级为ReAct模式 