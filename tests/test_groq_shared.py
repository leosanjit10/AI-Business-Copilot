"""Tests for the shared Groq completion handling."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from groq import APIError

from utils.shared import GroqCompletionError, generate_groq_text


class GroqCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.get_client = patch(
            "utils.shared.get_groq_client", return_value=self.client
        )
        self.get_client.start()
        self.addCleanup(self.get_client.stop)
        self.get_model = patch(
            "utils.shared.get_groq_model", return_value="test-model"
        )
        self.get_model.start()
        self.addCleanup(self.get_model.stop)

    def test_returns_current_sdk_message_content(self) -> None:
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="  Generated report  "),
                )
            ],
            usage=SimpleNamespace(completion_tokens=42),
            id="completion-id",
            _request_id="request-id",
        )

        result = generate_groq_text(
            [{"role": "user", "content": "Write a report"}],
            temperature=0.2,
        )

        self.assertEqual(result, "Generated report")
        self.client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Write a report"}],
            temperature=0.2,
        )

    def test_empty_content_raises_with_completion_diagnostics(self) -> None:
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(
                        content=None,
                        reasoning="reasoning was generated",
                        tool_calls=None,
                    ),
                )
            ],
            usage=SimpleNamespace(completion_tokens=123),
            id="completion-id",
            _request_id="request-id",
        )

        with self.assertLogs("utils.shared", level="ERROR"):
            with self.assertRaisesRegex(
                GroqCompletionError,
                r"finish_reason=length.*completion_tokens=123.*request_id=request-id",
            ):
                generate_groq_text([{"role": "user", "content": "Write a report"}])

    def test_empty_prompt_is_rejected_before_request(self) -> None:
        with self.assertRaisesRegex(GroqCompletionError, "prompt is empty"):
            generate_groq_text([{"role": "user", "content": "  "}])

        self.client.chat.completions.create.assert_not_called()

    def test_api_error_is_logged_and_exposed(self) -> None:
        api_error = APIError(
            "Invalid model supplied",
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
            body={"error": "model_not_found"},
        )
        self.client.chat.completions.create.side_effect = api_error

        with self.assertLogs("utils.shared", level="ERROR"):
            with self.assertRaisesRegex(
                GroqCompletionError, "Invalid model supplied"
            ):
                generate_groq_text([{"role": "user", "content": "Write a report"}])


if __name__ == "__main__":
    unittest.main()
