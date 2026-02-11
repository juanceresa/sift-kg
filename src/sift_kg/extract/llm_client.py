"""Multi-provider LLM client using LiteLLM.

Supports OpenAI, Anthropic, Ollama, and any other LiteLLM-compatible provider
with a single interface. Includes retry logic and cost tracking.
"""

import json
import logging
import re
import time

import litellm

logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


class LLMClient:
    """LLM client with retry logic and cost tracking."""

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        timeout: int = 120,
        system_message: str = "",
    ):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.system_message = system_message
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def call(self, prompt: str, system_message: str | None = None) -> str:
        """Call the LLM and return the text response.

        Args:
            prompt: The prompt to send
            system_message: Optional system message for context

        Returns:
            Response text

        Raises:
            RuntimeError: If all retries exhausted
        """
        last_error = None
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    timeout=self.timeout,
                    temperature=0.1,
                )

                # Track tokens and cost
                usage = response.usage
                if usage:
                    self.total_input_tokens += usage.prompt_tokens or 0
                    self.total_output_tokens += usage.completion_tokens or 0

                cost = litellm.completion_cost(completion_response=response)
                if cost:
                    self.total_cost_usd += cost

                text = response.choices[0].message.content or ""
                if not text.strip():
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return ""

                return text

            except litellm.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                last_error = "Rate limit exceeded"

            except litellm.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                last_error = "Request timed out"

            except Exception as e:
                logger.warning(f"LLM call failed: {e} (attempt {attempt + 1})")
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def call_json(self, prompt: str, system_message: str | None = None) -> dict:
        """Call LLM and parse the response as JSON.

        Handles common LLM quirks: markdown code fences, trailing text, etc.

        Args:
            prompt: The prompt (should request JSON output)
            system_message: Optional system message for context

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If response can't be parsed as JSON
        """
        text = self.call(prompt, system_message=system_message)
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
