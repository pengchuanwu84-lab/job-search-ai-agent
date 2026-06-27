from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "deepseek-v4-flash"


def get_configured_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def call_llm(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """最小大语言模型调用：OpenAI-compatible Chat Completions。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = get_configured_model()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {error.code}: {detail}") from error

    return data["choices"][0]["message"]


@dataclass
class Context:
    system_prompt: str
    max_messages: int = 12
    summary_limit: int = 4000
    summary: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.compact()

    def add_assistant(self, message: dict[str, Any]) -> None:
        saved = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("reasoning_content") is not None:
            saved["reasoning_content"] = message["reasoning_content"]
        if message.get("tool_calls"):
            saved["tool_calls"] = message["tool_calls"]
        self.messages.append(saved)

    def add_tool(self, tool_call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def build(self) -> list[dict[str, Any]]:
        system = self.system_prompt
        if self.summary:
            system += "\n\n已压缩的历史上下文：\n" + self.summary
        return [{"role": "system", "content": system}] + self.messages

    def compact(self) -> None:
        if len(self.messages) <= self.max_messages:
            return
        old = self.messages[:-self.max_messages]
        tail = self.messages[-self.max_messages :]
        while tail and tail[0]["role"] == "tool":
            old.append(tail.pop(0))
        block = "\n".join(_message_to_text(item) for item in old)
        self.summary = (self.summary + "\n" + block).strip()[-self.summary_limit :]
        self.messages = tail


def _message_to_text(message: dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    content = message.get("content") or ""
    if message.get("tool_calls"):
        names = [item.get("function", {}).get("name", "unknown") for item in message["tool_calls"]]
        content = f"tool_calls={names}"
    return f"{role}: {str(content)[:500]}"


def _safe_path(path: str) -> Path:
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"路径不允许越过项目目录: {path}")
    return resolved


def read_text(path: str) -> str:
    return _safe_path(path).read_text(encoding="utf-8")


def write_text(path: str, content: str) -> str:
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {target}"


TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "read_text": read_text,
    "write_text": write_text,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "读取项目内 UTF-8 文本文件。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_text",
            "description": "把文本写入项目内文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]


def run_tool(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function", {})
    name = function.get("name", "")
    raw_args = function.get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as error:
        return f"ERROR: invalid JSON arguments: {error}"
    if not isinstance(args, dict):
        return "ERROR: tool arguments must be a JSON object"
    if name not in TOOL_FUNCTIONS:
        return f"ERROR: unknown tool {name}"
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as error:  # noqa: BLE001
        return f"ERROR: {error}"


def run_agent(context: Context, user_input: str, max_steps: int = 6) -> str:
    context.add_user(user_input)
    for _ in range(max_steps):
        message = call_llm(context.build(), tools=TOOL_SCHEMAS)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            context.add_assistant(message)
            for tool_call in tool_calls:
                result = run_tool(tool_call)
                context.add_tool(tool_call["id"], result)
            continue

        content = message.get("content") or ""
        context.add_assistant({"content": content})
        context.compact()
        return content
    raise RuntimeError("工具调用超过max_steps，已停止")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    system_prompt: str

    def build_prompt(self, resume_path: str, jd_path: str, output_path: str) -> str:
        return f"""执行 {self.name}。

输入：
- resume_path: {resume_path}
- jd_path: {jd_path}
- output_path: {output_path}

步骤：
1. 调用 read_text 读取 resume_path。
2. 调用 read_text 读取 jd_path。
3. 只基于简历和JD输出求职分析，不编造经历。
4. 调用 write_text 保存 Markdown 报告到 output_path。
5. 最后用一句话返回已保存路径。

报告结构：
# 求职分析
## 岗位要求
## 简历匹配证据
## 简历改写建议
## 面试准备提示词
## 下一步行动
"""


JOB_SEARCH_SKILL = Skill(
    name="job_search_skill",
    description="读取简历和 JD，生成求职分析、简历改写建议和面试准备提示词。",
    system_prompt="""你是一个简洁的求职 Agent。
规则：
- 先看证据，再下结论。
- 不编造项目、公司、学历、成果数字。
- 输出短句，避免空话。
- 需要读写文件时必须调用工具。""",
)


GENERAL_CHAT_SYSTEM_PROMPT = """你是一个通用中文助手。
规则：
- 优先直接回答用户的普通问题。
- 回答简洁、具体，不编造不确定的事实。
- 需要读写项目文件时调用工具。
- 不猜测模型名称；模型名称由应用程序直接提供。"""

MODEL_QUERIES = {
    "/model",
    "当前模型",
    "模型名称",
    "你是什么模型",
    "你是哪个模型",
    "你用的什么模型",
    "你使用的是什么模型",
}


def is_model_query(user_input: str) -> bool:
    normalized = user_input.strip().lower().rstrip("?？。!！")
    return normalized in MODEL_QUERIES


def chat_loop() -> None:
    context = Context(system_prompt=GENERAL_CHAT_SYSTEM_PROMPT)
    print("通用对话已启动，输入 /model 查看模型，输入 exit、quit、q 或退出结束。")
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话已结束。")
            return

        if user_input.lower() in {"exit", "quit", "q", "退出"}:
            print("对话已结束。")
            return
        if not user_input:
            continue
        if is_model_query(user_input):
            print(f"Agent: 当前配置模型：{get_configured_model()}")
            continue

        try:
            reply = run_agent(context, user_input)
        except Exception as error:  # noqa: BLE001
            print(f"Agent 错误: {error}")
            continue
        print(f"Agent: {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="手搓最小 Agent：通用聊天 + 求职 Skill")
    parser.add_argument("--chat", action="store_true", help="启动连续对话模式")
    parser.add_argument("--resume", help="简历文件路径，项目根目录相对路径")
    parser.add_argument("--jd", help="JD 文件路径，项目根目录相对路径")
    parser.add_argument("--out", default="搭建ai agent（求职）/output/report.md", help="报告输出路径")
    args = parser.parse_args()

    if args.chat:
        if args.resume or args.jd:
            parser.error("--chat 不能与 --resume 或 --jd 同时使用")
        chat_loop()
        return
    if not args.resume or not args.jd:
        parser.error("非聊天模式必须同时提供 --resume 和 --jd")

    context = Context(system_prompt=JOB_SEARCH_SKILL.system_prompt)
    prompt = JOB_SEARCH_SKILL.build_prompt(args.resume, args.jd, args.out)
    print(run_agent(context, prompt))


if __name__ == "__main__":
    main()
