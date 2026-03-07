"""Thin OpenAI wrapper for structured LLM responses."""

import os
from typing import TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from science_bot.tracing import TraceWriter

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
T = TypeVar("T", bound=BaseModel)


class LLMProviderError(Exception):
    """Base exception for provider-layer failures."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        response_model_name: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        error_param: str | None = None,
        body: object | None = None,
    ) -> None:
        """Initialize a structured provider error."""

        super().__init__(message)
        self.message = message
        self.model = model
        self.response_model_name = response_model_name
        self.status_code = status_code
        self.request_id = request_id
        self.error_type = error_type
        self.error_code = error_code
        self.error_param = error_param
        self.body = body

    def __str__(self) -> str:
        """Render a concise provider error string."""

        details: list[str] = []
        if self.model is not None:
            details.append(f"model={self.model}")
        if self.response_model_name is not None:
            details.append(f"response_model={self.response_model_name}")
        if self.status_code is not None:
            details.append(f"status_code={self.status_code}")
        if self.request_id is not None:
            details.append(f"request_id={self.request_id}")
        if self.error_type is not None:
            details.append(f"error_type={self.error_type}")
        if self.error_code is not None:
            details.append(f"error_code={self.error_code}")
        if self.error_param is not None:
            details.append(f"error_param={self.error_param}")
        if not details:
            return self.message
        return f"{self.message} ({', '.join(details)})"


class LLMConfigurationError(LLMProviderError):
    """Raised when the LLM client is not properly configured."""


class LLMResponseFormatError(LLMProviderError):
    """Raised when a structured LLM response is missing or invalid."""


def _format_openai_error(
    exc: OpenAIError,
    *,
    model: str,
    response_model_name: str,
) -> LLMProviderError:
    """Build a structured provider error for an OpenAI failure."""

    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    body = getattr(exc, "body", None)
    error_type = None
    error_code = None
    error_param = None
    error_message = str(exc)

    if isinstance(body, dict):
        error_payload = body.get("error")
        if isinstance(error_payload, dict):
            error_type = error_payload.get("type")
            error_code = error_payload.get("code")
            error_param = error_payload.get("param")
            payload_message = error_payload.get("message")
            if isinstance(payload_message, str) and payload_message.strip():
                error_message = payload_message.strip()

    return LLMProviderError(
        f"OpenAI structured response request failed: {error_message}",
        model=model,
        response_model_name=response_model_name,
        status_code=status_code if isinstance(status_code, int) else None,
        request_id=request_id if isinstance(request_id, str) else None,
        error_type=error_type if isinstance(error_type, str) else None,
        error_code=error_code if isinstance(error_code, str) else None,
        error_param=error_param if isinstance(error_param, str) else None,
        body=body,
    )


async def parse_structured(
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    trace_writer: "TraceWriter | None" = None,
    trace_stage: str | None = None,
) -> T:
    """Request and parse a structured response into a Pydantic model.

    Args:
        system_prompt: Instruction text for the model.
        user_prompt: User input text to process.
        response_model: Pydantic model class to parse into.
        model: Optional model override.
        timeout_seconds: Request timeout in seconds.
        max_retries: SDK retry count for transient failures.
        trace_writer: Optional trace writer for prompt/response capture.
        trace_stage: Optional stage label used in trace events.

    Returns:
        T: Parsed structured response.

    Raises:
        LLMProviderError: If the provider call fails.
        LLMResponseFormatError: If the parsed response is missing or invalid.
    """
    resolved_model = model if model is not None else DEFAULT_OPENAI_MODEL
    if not resolved_model.strip():
        raise LLMConfigurationError("OpenAI model name must be non-empty.")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        raise LLMConfigurationError("OPENAI_API_KEY is required.")
    if timeout_seconds <= 0:
        raise LLMConfigurationError("timeout_seconds must be greater than zero.")
    if max_retries < 0:
        raise LLMConfigurationError("max_retries must be zero or greater.")

    if trace_writer is not None:
        trace_writer.write_event(
            event="llm_request",
            stage=trace_stage or "llm",
            payload={
                "model": resolved_model,
                "response_model": response_model.__name__,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
            },
        )

    client = AsyncOpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )

    try:
        response = await client.responses.parse(
            model=resolved_model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=response_model,
        )
    except OpenAIError as exc:
        provider_error = _format_openai_error(
            exc,
            model=resolved_model,
            response_model_name=response_model.__name__,
        )
        if trace_writer is not None:
            trace_writer.write_event(
                event="llm_error",
                stage=trace_stage or "llm",
                payload={
                    "type": type(exc).__name__,
                    "message": provider_error.message,
                    "model": provider_error.model,
                    "response_model": provider_error.response_model_name,
                    "status_code": provider_error.status_code,
                    "request_id": provider_error.request_id,
                    "error_type": provider_error.error_type,
                    "error_code": provider_error.error_code,
                    "error_param": provider_error.error_param,
                    "body": provider_error.body,
                },
            )
        raise provider_error from exc

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise LLMResponseFormatError(
            "Structured response did not include parsed output.",
            model=resolved_model,
            response_model_name=response_model.__name__,
        )
    if not isinstance(parsed, response_model):
        raise LLMResponseFormatError(
            "Structured response did not match the requested response model.",
            model=resolved_model,
            response_model_name=response_model.__name__,
        )

    if trace_writer is not None:
        trace_writer.write_event(
            event="llm_response",
            stage=trace_stage or "llm",
            payload={
                "model": resolved_model,
                "response_model": response_model.__name__,
                "parsed": parsed.model_dump(mode="python"),
            },
        )
    return parsed
