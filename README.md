# Job Search AI Agent

一个面向求职材料分析的最小 AI Agent 示例。程序读取 Markdown 格式的简历和岗位 JD，生成岗位匹配分析、简历修改建议、面试准备提示词和下一步行动，并将结果保存为 Markdown 报告。

## 技术栈

- Python
- DeepSeek API
- OpenAI-compatible Chat Completions
- Tool Calling
- Markdown

## 项目流程

```text
简历文件 + JD 文件 → Agent 分析 → Markdown 报告
```

Agent 通过 `read_text` 读取输入文件，通过 `write_text` 保存报告。分析提示词要求结论基于简历与 JD，不编造项目、公司、学历或成果。

## 运行方式

### 1. 环境要求

- Python 3.10 或更高版本
- 可用的 DeepSeek API Key

核心脚本只使用 Python 标准库，无需安装额外 Python 包。

### 2. 配置环境变量

请根据 `.env.example` 配置环境变量，不要提交真实 `.env` 或 API Key。

`.env.example` 示例：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
```

不要提交真实的 `.env` 或 API Key。

### 3. 运行示例分析

在仓库根目录执行：

```bash
REPO_DIR="$(pwd -W)"
python handmade_agent.py \
  --resume "$REPO_DIR/examples/resume_sample.md" \
  --jd "$REPO_DIR/examples/jd_sample.md" \
  --out "$REPO_DIR/output/report.md"
```

生成的报告位于 `output/report.md`。`output/` 已加入 `.gitignore`。

也可以启动连续对话模式：

```bash
python handmade_agent.py --chat
```

## 项目亮点

- 用较少代码展示完整 Agent 循环：模型调用、工具调用、工具结果回传和最终响应。
- 使用 OpenAI-compatible Chat Completions 接口，可通过环境变量配置服务地址和模型。
- 内置简历/JD 读取与 Markdown 报告写入工具。
- 对文件路径做项目目录范围检查，降低工具访问目录外文件的风险。
- 对较长对话保留最近消息，并压缩较早上下文。

## 示例与隐私

`examples/` 中的简历、JD 和报告均为虚构示例，不包含真实求职材料。上传前请阅读 `REVIEW_BEFORE_UPLOAD.md`，并再次确认未加入真实简历、真实 JD、真实报告、API Key 或 `.env`。
