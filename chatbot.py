from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_MAX_HISTORY_MESSAGES = 10
SUPPORTED_LLM_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5",
    "deepseek/deepseek-chat",
]

EMOTION_PROMPT_RULES = {
    "anger": (
        "The driver sounds angry. Help them slow down emotionally, use a calm tone, "
        "and briefly warn against aggressive driving."
    ),
    "fear": (
        "The driver sounds afraid. Reassure them briefly and, when appropriate, ask "
        "whether they want to pull over somewhere safe."
    ),
    "sad": (
        "The driver sounds sad. Show brief empathy, check on their state, and avoid "
        "sounding overly cheerful."
    ),
    "happy": (
        "The driver sounds positive. Keep the tone natural, concise, and attentive to safety."
    ),
    "neutral": (
        "The driver sounds neutral. Keep the tone practical, calm, and concise."
    ),
    "disgust": (
        "The driver sounds frustrated or uncomfortable. Keep the tone calm and redirect "
        "attention to steady driving."
    ),
    "surprise": (
        "The driver sounds startled. Keep the tone calm and help them refocus on the road."
    ),
}


@dataclass
class ChatbotResponse:
    text: str
    model: str
    emotion: str
    temperature: float
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def trim_history(
    conversation_history: list[dict[str, str]] | None,
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
) -> list[dict[str, str]]:
    if not conversation_history:
        return []

    filtered_history = [
        message
        for message in conversation_history
        if message.get("role") in {"user", "assistant"}
    ]
    return filtered_history[-max_messages:]


def build_system_prompt(emotion: str, auto_trigger: bool = False) -> str:
    normalized_emotion = emotion.strip().lower() if emotion else "neutral"
    emotion_rule = EMOTION_PROMPT_RULES.get(
        normalized_emotion,
        "Adjust tone gently to the driver's emotional state and keep them focused on safe driving.",
    )
    auto_rule = (
        "No driver message was provided. Proactively send one short check-in that sounds natural."
        if auto_trigger
        else "Respond directly to the driver's message."
    )
    return (
        "You are an in-car emotional support assistant for a driver.\n"
        "Rules:\n"
        "- Reply in at most 2 short sentences, and never exceed 3 sentences.\n"
        "- Stay calm, grounded, and non-alarmist.\n"
        "- Do not produce long explanations or lists.\n"
        "- Keep the driver focused; avoid distracting follow-up questions unless necessary.\n"
        "- If the situation sounds unsafe, gently encourage safer behavior or pulling over.\n"
        f"- Current detected emotion: {normalized_emotion}.\n"
        f"- Emotion guidance: {emotion_rule}\n"
        f"- Interaction mode: {auto_rule}"
    )


class DriverAssistantChatbot:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        app_title: str = "Driver Assistant Chatbot",
        referer: str | None = None,
    ) -> None:
        load_dotenv()
        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY was not found. Set it in the environment or .env first."
            )

        default_headers = {"X-Title": app_title}
        if referer or os.getenv("OPENROUTER_HTTP_REFERER"):
            default_headers["HTTP-Referer"] = referer or os.getenv(
                "OPENROUTER_HTTP_REFERER", ""
            )

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    def generate_reply(
        self,
        emotion: str,
        user_message: str | None,
        model: str = DEFAULT_MODEL,
        temperature: float = 1.0,
        conversation_history: list[dict[str, str]] | None = None,
        max_output_tokens: int = 120,
        auto_trigger: bool = False,
    ) -> ChatbotResponse:
        system_prompt = build_system_prompt(emotion, auto_trigger=auto_trigger)
        trimmed_history = trim_history(conversation_history)
        current_user_message = (
            user_message.strip()
            if user_message
            else (
                "No user message is available. Provide a brief, calm proactive check-in."
                if auto_trigger
                else "Respond briefly and safely."
            )
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *trimmed_history,
            {"role": "user", "content": current_user_message},
        ]

        start = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        content = completion.choices[0].message.content or ""
        usage = completion.usage
        return ChatbotResponse(
            text=content.strip(),
            model=model,
            emotion=emotion,
            temperature=temperature,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple CLI for the OpenRouter driver assistant chatbot."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenRouter model id.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        help="Current detected emotion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=None,
        help="Optional JSON file with prior user/assistant messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chatbot = DriverAssistantChatbot()
    history: list[dict[str, str]] = []
    if args.history_json and args.history_json.exists():
        history = json.loads(args.history_json.read_text(encoding="utf-8"))

    print("Driver assistant chatbot CLI. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_text = input("Driver: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        response = chatbot.generate_reply(
            emotion=args.emotion,
            user_message=user_text,
            model=args.model,
            temperature=args.temperature,
            conversation_history=history,
        )
        print(f"Assistant: {response.text}\n")
        history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response.text},
            ]
        )
        print(json.dumps(asdict(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
