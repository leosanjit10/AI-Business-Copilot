"""Shared environment, session-state, and Groq helpers for Streamlit pages."""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import APIError, Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)

SESSION_DATA_KEYS = (
    "df",
    "daily_sales",
    "product_sales",
    "total_revenue",
    "total_orders",
    "best_product",
    "average_order_value",
)


class GroqCompletionError(RuntimeError):
    """Raised when a Groq request cannot produce validated assistant text."""


def apply_compact_page_spacing() -> None:
    """Apply a shared, compact vertical rhythm to Streamlit page content."""
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.25rem !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
            gap: .85rem;
        }
        [data-testid="stMainBlockContainer"] h1 {
            margin-top: .1rem;
            margin-bottom: .2rem;
        }
        div[data-testid="stMetric"] {
            min-height: 112px;
            height: 100%;
            justify-content: center;
            padding: .8rem .9rem;
            border: 1px solid rgba(133, 164, 197, .25);
            border-left: 3px solid #42d6c5;
            border-radius: 15px;
            background: linear-gradient(145deg, rgba(24, 42, 65, .98), rgba(15, 29, 47, .98));
            box-shadow: 0 10px 26px rgba(0, 0, 0, .2);
            transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: rgba(102, 224, 207, .65);
            box-shadow: 0 16px 32px rgba(0, 0, 0, .28);
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #c2cfdf !important;
            font-size: .78rem !important;
            font-weight: 650 !important;
            line-height: 1.25;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f7faff !important;
            font-size: clamp(1.65rem, 2.45vw, 2.45rem) !important;
            font-weight: 780 !important;
            letter-spacing: -.045em;
            line-height: 1.1;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-size: .78rem;
            font-weight: 650;
        }
        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: .9rem !important;
                padding-bottom: 1.5rem !important;
            }
            [data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
                gap: .7rem;
            }
            div[data-testid="stMetric"] {
                min-height: 92px;
                padding: .65rem .7rem;
            }
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: clamp(1.4rem, 6vw, 1.85rem) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_prompt_card() -> None:
    """Show the same concise next step wherever forecast data is absent."""
    with st.container(border=True):
        st.markdown("**:material/insights: Forecast insights**")
        st.caption(
            "Generate a forecast from the Forecasting page to enable forecast insights."
        )


def require_dashboard_data() -> bool:
    """Show a helpful prompt and return False if dashboard data is unavailable."""
    missing = [
        key
        for key in SESSION_DATA_KEYS
        if key not in st.session_state or st.session_state[key] is None
    ]
    dataframe_keys = ("df", "daily_sales", "product_sales")
    invalid_dataframes = [
        key
        for key in dataframe_keys
        if key in st.session_state
        and (
            not isinstance(st.session_state[key], pd.DataFrame)
            or st.session_state[key].empty
        )
    ]
    if missing or invalid_dataframes:
        st.warning("Upload a valid sales CSV on the Dashboard to use this page.")
        return False
    return True


def get_groq_client() -> Groq:
    """Create a Groq client using the project .env or process environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        st.error(
            "GROQ_API_KEY is missing or empty. Add it to the project .env file "
            "and restart Streamlit."
        )
        st.stop()
    return Groq(api_key=api_key)


def get_groq_model() -> str:
    """Return the configured Groq model, with the current app default."""
    default_model = "openai/gpt-oss-20b"
    return os.getenv("GROQ_MODEL", default_model).strip() or default_model


def generate_groq_text(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
) -> str:
    """Request a chat completion and return validated assistant text."""
    if not messages or any(
        not isinstance(message.get("content"), str)
        or not message["content"].strip()
        for message in messages
    ):
        raise GroqCompletionError(
            "The Groq request was not sent because its prompt is empty."
        )

    model = get_groq_model()
    prompt_chars = sum(len(message["content"]) for message in messages)
    request_options: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        request_options["temperature"] = temperature

    try:
        response = get_groq_client().chat.completions.create(**request_options)
    except APIError as error:
        logger.exception(
            "Groq API request failed (model=%s, status=%s, request_id=%s, "
            "prompt_chars=%s)",
            model,
            getattr(error, "status_code", None),
            getattr(error, "request_id", None),
            prompt_chars,
        )
        raise GroqCompletionError(f"Groq API request failed: {error}") from error

    request_id = getattr(response, "_request_id", None)
    response_id = getattr(response, "id", None)
    choices = getattr(response, "choices", None)
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)

    if isinstance(content, str) and content.strip():
        logger.info(
            "Groq completion returned text (model=%s, request_id=%s, "
            "response_id=%s, finish_reason=%s, prompt_chars=%s, "
            "completion_tokens=%s)",
            model,
            request_id,
            response_id,
            finish_reason,
            prompt_chars,
            completion_tokens,
        )
        return content.strip()

    diagnostics = (
        f"model={model}, finish_reason={finish_reason or 'unknown'}, "
        f"completion_tokens={completion_tokens if completion_tokens is not None else 'unknown'}, "
        f"request_id={request_id or response_id or 'unavailable'}, "
        f"reasoning_present={bool(getattr(message, 'reasoning', None))}, "
        f"tool_calls_present={bool(getattr(message, 'tool_calls', None))}"
    )
    logger.error(
        "Groq returned no assistant text (%s, prompt_chars=%s, content_type=%s)",
        diagnostics,
        prompt_chars,
        type(content).__name__ if content is not None else "None",
    )
    raise GroqCompletionError(
        "Groq returned no assistant text. "
        f"Completion diagnostics: {diagnostics}."
    )
