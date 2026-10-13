"""Chat with the assistant in the terminal.

uv run chat                               # a conversation; /new starts over, /quit (or Ctrl-D) ends it
uv run chat "週五用街口支付有什麼優惠?"      # one question, one answer
uv run chat --tools                       # also show each tool call
uv run chat --today 2026-10-10            # pretend it's that day (the golden set's date)
"""

import argparse
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import date

from langchain_core.messages import AIMessage, BaseMessage

from promo_rag.agents import AgentError
from promo_rag.agents.tools.dates import pinned_today, today
from promo_rag.agents.tracing import flush_traces
from promo_rag.workflows.chat.assistant import ask, transcript

HELP = "Ask about POYA's promotions and stores. /new starts over, /quit ends."
DIM, RESET = "\033[2m", "\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(prog="chat")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("--tools", action="store_true", help="show each tool call")
    parser.add_argument("--today", type=date.fromisoformat, help="pin today's date (YYYY-MM-DD)")
    args = parser.parse_args()

    with pinned_today(args.today) if args.today else nullcontext():
        if args.question:
            ok = _turn(args.question, [], str(uuid.uuid4()), args.tools) is not None
            flush_traces()
            return 0 if ok else 1
        try:
            import readline  # noqa: F401  (arrow keys and line editing for input())
        except ImportError:
            pass
        print(f"{HELP}\nToday is {today():%Y-%m-%d (%A)}.\n")
        history: list[BaseMessage] = []
        session_id = str(uuid.uuid4())
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question in ("/quit", "/exit"):
                break
            if question == "/new":
                history, session_id = [], str(uuid.uuid4())
                print("Started a new conversation.\n")
                continue
            if question.startswith("/"):
                print(HELP + "\n")
                continue
            messages = _turn(question, history, session_id, args.tools)
            if messages is not None:
                history = transcript(messages)
    flush_traces()
    return 0


def _turn(
    question: str, history: list[BaseMessage], session_id: str, show_tools: bool
) -> list[BaseMessage] | None:
    """Ask, print the answer, and return the conversation (None if the call failed)."""
    started = time.monotonic()
    try:
        result = ask(question, history=history, session_id=session_id)
    except AgentError as e:
        print(f"\nError: the model call failed ({e}). Try again.\n", file=sys.stderr)
        return None
    turn = result.messages[len(history) + 1 :]
    calls = [c for m in turn if isinstance(m, AIMessage) for c in m.tool_calls]
    if show_tools:
        for c in calls:
            args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
            print(f"{DIM}  → {c['name']}({args}){RESET}")
    seconds = time.monotonic() - started
    print(f"\n{result.text}\n")
    print(f"{DIM}({seconds:.1f} s, {len(calls)} tool call(s), {result.model}){RESET}\n")
    return result.messages


if __name__ == "__main__":
    sys.exit(main())
