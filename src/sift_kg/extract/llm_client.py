"""Multi-provider LLM client using LiteLLM.

Supports OpenAI, Anthropic, Ollama, and any other LiteLLM-compatible provider
with a single interface. Includes retry logic, cost tracking, async support,
and a token-bucket rate limiter to avoid burning through API credits.
"""

import asyncio
import collections
import json
import logging
import re
import time

import litellm

logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


class _RateLimiter:
    """Sliding-window rate limiter. Tracks call timestamps and sleeps
    before issuing a call that would exceed the RPM budget.

    Thread-safe for sync calls via time.sleep.
    Async-safe via asyncio.Lock + asyncio.sleep.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._window = 60.0  # seconds
        self._timestamps: collections.deque[float] = collections.deque()
        self._async_lock = asyncio.Lock()

    def wait_sync(self) -> None:
        """Block until we can make a call without exceeding RPM."""
        if self.rpm <= 0:
            return
        now = time.monotonic()
        self._purge(now)
        if len(self._timestamps) >= self.rpm:
            oldest = self._timestamps[0]
            sleep_for = self._window - (now - oldest) + 0.1
            if sleep_for > 0:
                logger.debug(f"Rate limiter: sleeping {sleep_for:.1f}s ({self.rpm} RPM)")
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())

    async def wait_async(self) -> None:
        """Async version — serializes the check so concurrent tasks don't
        all decide they can go at once."""
        if self.rpm <= 0:
            return
        async with self._async_lock:
            now = time.monotonic()
            self._purge(now)
            if len(self._timestamps) >= self.rpm:
                oldest = self._timestamps[0]
                sleep_for = self._window - (now - oldest) + 0.1
                if sleep_for > 0:
                    logger.debug(f"Rate limiter: sleeping {sleep_for:.1f}s ({self.rpm} RPM)")
                    await asyncio.sleep(sleep_for)
            self._timestamps.append(time.monotonic())

    def _purge(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class LLMClient:
    """LLM client with retry logic, cost tracking, and rate limiting."""

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        rate_limit_retries: int = 8,
        rate_limit_base_wait: float = 5.0,
        rpm: int = 40,
        timeout: int = 120,
        system_message: str = "",
    ):
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_base_wait = rate_limit_base_wait
        self.timeout = timeout
        self.system_message = system_message
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._limiter = _RateLimiter(rpm)

    def _build_messages(self, prompt: str, system_message: str | None) -> list[dict]:
        effective_system = system_message or self.system_message
        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _track_usage(self, response: object) -> None:
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens or 0
            self.total_output_tokens += usage.completion_tokens or 0
        cost = litellm.completion_cost(completion_response=response)
        if cost:
            self.total_cost_usd += cost

    def call(self, prompt: str, system_message: str | None = None) -> str:
        """Call the LLM and return the text response."""
        last_error = None
        messages = self._build_messages(prompt, system_message)

        rate_limit_hits = 0
        error_retries = 0

        while error_retries < self.max_retries:
            self._limiter.wait_sync()
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    timeout=self.timeout,
                    temperature=0.1,
                )
                self._track_usage(response)

                text = response.choices[0].message.content or ""
                if not text.strip():
                    error_retries += 1
                    logger.warning(f"Empty response (attempt {error_retries}/{self.max_retries})")
                    if error_retries < self.max_retries:
                        time.sleep(1)
                        continue
                    raise RuntimeError(
                        f"LLM returned empty response after {self.max_retries} attempts"
                    )

                return text

            except litellm.RateLimitError as exc:
                rate_limit_hits += 1
                if rate_limit_hits > self.rate_limit_retries:
                    raise RuntimeError(
                        f"Rate limited {rate_limit_hits} times, giving up. "
                        f"Consider upgrading to a paid tier or using a different model."
                    ) from exc
                wait = min(self.rate_limit_base_wait * (2 ** (rate_limit_hits - 1)), 60)
                logger.warning(
                    f"Rate limited, waiting {wait:.0f}s "
                    f"(rate limit hit {rate_limit_hits}/{self.rate_limit_retries})"
                )
                time.sleep(wait)
                last_error = "Rate limit exceeded"

            except litellm.Timeout:
                error_retries += 1
                logger.warning(f"Timeout (attempt {error_retries}/{self.max_retries})")
                last_error = "Request timed out"

            except KeyboardInterrupt:
                raise
            except Exception as e:
                error_retries += 1
                logger.warning(f"LLM call failed: {e} (attempt {error_retries}/{self.max_retries})")
                last_error = str(e)
                if error_retries < self.max_retries:
                    time.sleep(1)

        raise RuntimeError(f"LLM call failed after {error_retries} attempts: {last_error}")

    def call_json(self, prompt: str, system_message: str | None = None) -> dict:
        """Call LLM and parse the response as JSON."""
        text = self.call(prompt, system_message=system_message)
        return parse_llm_json(text)

    async def acall(self, prompt: str, system_message: str | None = None) -> str:
        """Async version of call(). Rate-limited via shared token bucket."""
        last_error = None
        messages = self._build_messages(prompt, system_message)

        rate_limit_hits = 0
        error_retries = 0

        while error_retries < self.max_retries:
            await self._limiter.wait_async()
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    timeout=self.timeout,
                    temperature=0.1,
                )
                self._track_usage(response)

                text = response.choices[0].message.content or ""
                if not text.strip():
                    error_retries += 1
                    logger.warning(f"Empty response (attempt {error_retries}/{self.max_retries})")
                    if error_retries < self.max_retries:
                        await asyncio.sleep(1)
                        continue
                    raise RuntimeError(
                        f"LLM returned empty response after {self.max_retries} attempts"
                    )

                return text

            except litellm.RateLimitError as exc:
                rate_limit_hits += 1
                if rate_limit_hits > self.rate_limit_retries:
                    raise RuntimeError(
                        f"Rate limited {rate_limit_hits} times, giving up. "
                        f"Consider upgrading to a paid tier or using a different model."
                    ) from exc
                wait = min(self.rate_limit_base_wait * (2 ** (rate_limit_hits - 1)), 60)
                logger.warning(
                    f"Rate limited, waiting {wait:.0f}s "
                    f"(rate limit hit {rate_limit_hits}/{self.rate_limit_retries})"
                )
                await asyncio.sleep(wait)
                last_error = "Rate limit exceeded"

            except asyncio.CancelledError:
                raise
            except litellm.Timeout:
                error_retries += 1
                logger.warning(f"Timeout (attempt {error_retries}/{self.max_retries})")
                last_error = "Request timed out"

            except KeyboardInterrupt:
                raise
            except Exception as e:
                error_retries += 1
                logger.warning(f"LLM call failed: {e} (attempt {error_retries}/{self.max_retries})")
                last_error = str(e)
                if error_retries < self.max_retries:
                    await asyncio.sleep(1)

        raise RuntimeError(f"LLM call failed after {error_retries} attempts: {last_error}")

    async def acall_json(self, prompt: str, system_message: str | None = None) -> dict:
        """Async version of call_json()."""
        text = await self.acall(prompt, system_message=system_message)
        return parse_llm_json(text)


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, handling common quirks.

    LLMs often wrap JSON in markdown code fences or include
    trailing explanation text. This function strips that.
    """
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a valid JSON object by scanning for balanced braces
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        break
            break

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")
