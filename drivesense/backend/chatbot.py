from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from drivesense.backend.speech import detect_text_language, is_supported_zh_en_text


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
logger = logging.getLogger(__name__)
DEFAULT_MODEL = "openai/gpt-4o-mini"
FALLBACK_MODEL = "deepseek/deepseek-chat"
DEFAULT_MAX_HISTORY_MESSAGES = 10
SUPPORTED_LLM_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5",
    "deepseek/deepseek-chat",
]
ALLOWED_DRIVER_STATE_KEYS = (
    "driver_detected",
    "driver_confident",
    "emotion",
    "emotion_confidence",
    "emotion_secondary",
    "emotion_secondary_confidence",
    "eye_label",
    "eye_confidence",
    "risk",
    "focus_alert",
    "driver_side",
    "closed_eye_duration",
    "focus_level",
    "trigger_reason",
)

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
    selected_model: str
    emotion: str
    temperature: float
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    fallback_used: bool


def format_driver_state(driver_state: dict[str, Any] | None) -> str:
    sanitized_state = sanitize_driver_state(driver_state)
    if not sanitized_state:
        return "No structured driver state is available."

    emotion = str(sanitized_state.get("emotion", "neutral"))
    emotion_conf = float(sanitized_state.get("emotion_confidence", 0.0))
    emotion_secondary = str(sanitized_state.get("emotion_secondary", ""))
    emotion_secondary_conf = float(sanitized_state.get("emotion_secondary_confidence", 0.0))
    eye_label = str(sanitized_state.get("eye_label", "open_eye"))
    eye_conf = float(sanitized_state.get("eye_confidence", 0.0))
    risk = str(sanitized_state.get("risk", "OK"))
    focus_alert = bool(sanitized_state.get("focus_alert", False))
    driver_detected = bool(sanitized_state.get("driver_detected", False))
    closed_eye_duration = float(sanitized_state.get("closed_eye_duration", 0.0))
    trigger_reason = str(sanitized_state.get("trigger_reason", "")).strip()
    return (
        f"DriverDetected={'yes' if driver_detected else 'no'}, "
        f"TopEmotions={emotion} ({emotion_conf:.2f})"
        + (
            f", {emotion_secondary} ({emotion_secondary_conf:.2f})"
            if emotion_secondary
            else ""
        )
        + ", "
        f"Eyes={eye_label} ({eye_conf:.2f}), "
        f"Risk={risk}, "
        f"FocusAlert={'yes' if focus_alert else 'no'}, "
        f"ClosedEyeDuration={closed_eye_duration:.1f}s"
        + (f", TriggerReason={trigger_reason}." if trigger_reason else ".")
    )


def sanitize_driver_state(driver_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not driver_state:
        return None
    return {
        key: driver_state[key]
        for key in ALLOWED_DRIVER_STATE_KEYS
        if key in driver_state
    }


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


def build_system_prompt(
    emotion: str,
    auto_trigger: bool = False,
    driver_state: dict[str, Any] | None = None,
    provider_safe: bool = False,
    response_language: str = "en",
) -> str:
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
    language_rule = (
        "Reply only in Simplified Chinese."
        if response_language == "zh"
        else "Reply only in English."
    )
    if provider_safe:
        return (
            "You are a brief conversational assistant inside a desktop application.\n"
            "Rules:\n"
            "- Reply in at most 2 short sentences, and never exceed 3 sentences.\n"
            "- Stay calm, practical, and non-alarmist.\n"
            "- Do not produce long explanations or lists.\n"
            "- Treat app-provided state as uncertain context, not ground truth.\n"
            "- Avoid medical, legal, or high-stakes operational instructions.\n"
            f"- {language_rule}\n"
            f"- App emotion context: {normalized_emotion}.\n"
            f"- App state context: {format_driver_state(driver_state)}\n"
            f"- Tone guidance: {emotion_rule}\n"
            f"- Interaction mode: {auto_rule}"
        )

    return (
        "You are a brief wellbeing and focus-support assistant inside a desktop application.\n"
        "Rules:\n"
        "- Reply in at most 2 short sentences, and never exceed 3 sentences.\n"
        "- Stay calm, grounded, and non-alarmist.\n"
        "- Do not produce long explanations or lists.\n"
        "- Use the app signals as soft context only.\n"
        "- Help the user stay calm and focused without sounding forceful.\n"
        f"- {language_rule}\n"
        f"- Current emotion context: {normalized_emotion}.\n"
        f"- Current app state: {format_driver_state(driver_state)}\n"
        f"- Tone guidance: {emotion_rule}\n"
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
        driver_state: dict[str, Any] | None = None,
    ) -> ChatbotResponse:
        sanitized_driver_state = sanitize_driver_state(driver_state)
        response_language = "zh" if detect_text_language(user_message or "") == "zh" else "en"
        system_prompt = build_system_prompt(
            emotion,
            auto_trigger=auto_trigger,
            driver_state=sanitized_driver_state,
            response_language=response_language,
        )
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

        def attempt_completion(
            active_model: str,
            active_system_prompt: str,
            provider_safe: bool,
        ) -> Any:
            active_messages: list[dict[str, Any]] = [
                {"role": "system", "content": active_system_prompt},
                *trimmed_history,
                {"role": "user", "content": current_user_message},
            ]
            logger.info(
                "OpenRouter request start | model=%s temperature=%.2f emotion=%s auto_trigger=%s provider_safe=%s driver_state=%s",
                active_model,
                temperature,
                emotion,
                auto_trigger,
                provider_safe,
                format_driver_state(sanitized_driver_state),
            )
            start = time.perf_counter()
            try:
                result = self.client.chat.completions.create(
                    model=active_model,
                    messages=active_messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                latency = (time.perf_counter() - start) * 1000.0
                return result, latency
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000.0
                logger.exception(
                    "OpenRouter request failed | model=%s temperature=%.2f latency_ms=%.0f provider_safe=%s",
                    active_model,
                    temperature,
                    latency,
                    provider_safe,
                )
                raise RuntimeError(
                    f"OpenRouter request failed for model '{active_model}': {exc}"
                ) from exc

        used_model = model
        try:
            completion, latency_ms = attempt_completion(
                used_model,
                system_prompt,
                provider_safe=False,
            )
        except RuntimeError as exc:
            message = str(exc)
            should_retry_provider_safe = (
                "403" in message
                and ("openai/" in model or "anthropic/" in model)
            )
            if not should_retry_provider_safe:
                raise
            safe_prompt = build_system_prompt(
                emotion,
                auto_trigger=auto_trigger,
                driver_state=sanitized_driver_state,
                provider_safe=True,
                response_language=response_language,
            )
            logger.warning(
                "Retrying OpenRouter request with provider-safe prompt | model=%s",
                model,
            )
            try:
                completion, latency_ms = attempt_completion(
                    used_model,
                    safe_prompt,
                    provider_safe=True,
                )
            except RuntimeError as safe_exc:
                safe_message = str(safe_exc)
                should_fallback_model = (
                    "403" in safe_message
                    and used_model != FALLBACK_MODEL
                )
                if not should_fallback_model:
                    raise
                used_model = FALLBACK_MODEL
                logger.warning(
                    "Falling back to alternative model after provider rejection | fallback_model=%s",
                    used_model,
                )
                fallback_prompt = build_system_prompt(
                    emotion,
                    auto_trigger=auto_trigger,
                    driver_state=sanitized_driver_state,
                    provider_safe=True,
                    response_language=response_language,
                )
                completion, latency_ms = attempt_completion(
                    used_model,
                    fallback_prompt,
                    provider_safe=True,
                )

        content = (completion.choices[0].message.content or "").strip()
        if content and not is_supported_zh_en_text(content):
            logger.warning(
                "Model returned unsupported language content | model=%s text=%r",
                used_model,
                content,
            )
            content = (
                "Please stay calm and keep your attention on the road."
                if response_language == "en"
                else "请保持冷静，专注前方道路。"
            )
        if not content.strip():
            logger.warning(
                "OpenRouter returned empty content | model=%s latency_ms=%.0f",
                used_model,
                latency_ms,
            )
        usage = completion.usage
        logger.info(
            "OpenRouter request success | model=%s latency_ms=%.0f prompt_tokens=%s completion_tokens=%s",
            used_model,
            latency_ms,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return ChatbotResponse(
            text=content.strip(),
            model=used_model,
            selected_model=model,
            emotion=emotion,
            temperature=temperature,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            fallback_used=used_model != model,
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
