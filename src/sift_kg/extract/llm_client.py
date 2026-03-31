"""Multi-provider LLM client using LiteLLM.

Supports OpenAI, Anthropic, Ollama, and any other LiteLLM-compatible provider
with a single interface. Includes retry logic, cost tracking, async support,
and a token-bucket rate limiter to avoid burning through API credits.

## 与常见 llm_client 设计的区别：

1. **专注于知识图谱提取场景**：不是通用LLM客户端，专门为批量文档提取任务优化设计
2. **内置滑动窗口限速**：自带令牌桶限流器，避免批量并发调用触发API提供商限流，不需要外部协调
3. **指数退避重试**：对限流错误单独处理，使用指数退避策略，最大重试次数可配置
4. **完整成本追踪**：全程追踪每个调用的token消耗和美元成本，支持`--max-cost`预算控制功能
5. **同步/异步双接口**：同时提供同步和异步调用接口，异步版本利用信号量控制并发，适合批量处理
6. **鲁棒的JSON解析**：专门处理LLM输出JSON的常见问题（markdown代码块包裹、多余文本前后缀），通过扫描平衡括号尝试恢复有效JSON
7. **低温度默认设置**：默认temperature=0.1，适合提取这类需要确定性输出的任务，而不是创作
"""

import asyncio
import collections
import json
import logging
import re
import time

import litellm

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# 禁用LiteLLM的冗长日志和OpenAI重试 chatter，避免日志污染
litellm.suppress_debug_info = True
litellm.set_verbose = False
for _name in (
    "LiteLLM",
    "litellm",
    "LiteLLM Proxy",
    "LiteLLM Router",
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
):
    # 将这些模块的日志级别设为WARNING，只输出警告及以上信息
    logging.getLogger(_name).setLevel(logging.WARNING)


class _RateLimiter:
    """滑动窗口限流器。

    跟踪调用时间戳，在调用会超出RPM（每分钟请求数）预算时休眠。

    通过time.sleep实现同步调用线程安全。
    通过asyncio.Lock + asyncio.sleep实现异步安全。
    """

    def __init__(self, rpm: int):
        # 每分钟允许的请求数
        self.rpm = rpm
        # 滑动窗口大小（秒），固定为60秒
        self._window = 60.0
        # 存储已发生请求的时间戳队列
        self._timestamps: collections.deque[float] = collections.deque()
        # 异步锁，保护并发检查
        self._async_lock = asyncio.Lock()

    def wait_sync(self) -> None:
        """阻塞直到可以发起调用而不超过RPM限制。"""
        # RPM <= 0 表示不限速，直接返回
        if self.rpm <= 0:
            return
        # 获取当前单调时间（不受系统时间改变影响）
        now = time.monotonic()
        # 清除窗口外的过期时间戳
        self._purge(now)
        # 如果已在窗口内的请求数大于等于限制，需要等待
        if len(self._timestamps) >= self.rpm:
            # 获取最老的时间戳
            oldest = self._timestamps[0]
            # 计算需要等待的时间：加上0.1秒额外缓冲
            sleep_for = self._window - (now - oldest) + 0.1
            if sleep_for > 0:
                logger.debug(f"限流器：休眠 {sleep_for:.1f}s ({self.rpm} RPM)")
                time.sleep(sleep_for)
        # 添加当前时间戳到队列
        self._timestamps.append(time.monotonic())

    async def wait_async(self) -> None:
        """异步版本 —— 序列化检查，避免并发任务同时判断都通过。"""
        # RPM <= 0 表示不限速，直接返回
        if self.rpm <= 0:
            return
        # 使用异步锁保证同时只有一个任务检查
        async with self._async_lock:
            now = time.monotonic()
            # 清除窗口外的过期时间戳
            self._purge(now)
            # 如果已在窗口内的请求数大于等于限制，需要等待
            if len(self._timestamps) >= self.rpm:
                oldest = self._timestamps[0]
                sleep_for = self._window - (now - oldest) + 0.1
                if sleep_for > 0:
                    logger.debug(f"限流器：休眠 {sleep_for:.1f}s ({self.rpm} RPM)")
                    await asyncio.sleep(sleep_for)
            # 添加当前时间戳到队列
            self._timestamps.append(time.monotonic())

    def _purge(self, now: float) -> None:
        """清除滑动窗口外的过期时间戳。"""
        # 计算截止时间
        cutoff = now - self._window
        # 弹出所有早于截止时间的时间戳
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class LLMClient:
    """带有重试逻辑、成本追踪和限速的LLM客户端。"""

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
        # 使用的模型标识符（LiteLLM格式，如openai/gpt-4o-mini）
        self.model = model
        # 普通错误最大重试次数
        self.max_retries = max_retries
        # 限流错误最大重试次数
        self.rate_limit_retries = rate_limit_retries
        # 限流错误指数退避的基础等待时间（秒）
        self.rate_limit_base_wait = rate_limit_base_wait
        # 请求超时时间（秒）
        self.timeout = timeout
        # 默认系统提示词
        self.system_message = system_message
        # 累计总成本（美元）
        self.total_cost_usd = 0.0
        # 累计输入token数
        self.total_input_tokens = 0
        # 累计输出token数
        self.total_output_tokens = 0
        # 限流器实例
        self._limiter = _RateLimiter(rpm)

    def _build_messages(self, prompt: str, system_message: str | None) -> list[dict]:
        """构建LLM消息列表。"""
        # 使用传入的系统提示词，如果为None则使用默认
        effective_system = system_message or self.system_message
        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _track_usage(self, response: object) -> None:
        """追踪token使用量和成本。"""
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens or 0
            self.total_output_tokens += usage.completion_tokens or 0
        # 使用LiteLLM计算本次调用成本
        cost = litellm.completion_cost(completion_response=response)
        if cost:
            self.total_cost_usd += cost

    def call(self, prompt: str, system_message: str | None = None) -> str:
        """同步调用LLM并返回文本响应。"""
        last_error = None
        messages = self._build_messages(prompt, system_message)

        # 限流命中计数
        rate_limit_hits = 0
        # 普通错误重试计数
        error_retries = 0

        while error_retries < self.max_retries:
            # 等待限速器允许调用
            self._limiter.wait_sync()
            try:
                # 调用LiteLLM完成请求
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    timeout=self.timeout,
                    temperature=0.1,
                )
                # 追踪使用量和成本
                self._track_usage(response)

                # 提取响应文本
                text = response.choices[0].message.content or ""
                if not text.strip():
                    # 空响应，重试
                    error_retries += 1
                    logger.warning(f"空响应 (尝试 {error_retries}/{self.max_retries})")
                    if error_retries < self.max_retries:
                        time.sleep(1)
                        continue
                    # 达到最大重试次数，抛出异常
                    raise RuntimeError(f"LLM在{self.max_retries}次尝试后返回空响应")

                return text

            except litellm.RateLimitError as exc:
                # 限流错误，单独处理重试计数
                rate_limit_hits += 1
                if rate_limit_hits > self.rate_limit_retries:
                    # 达到限流重试上限，放弃
                    raise RuntimeError(
                        f"被限流{rate_limit_hits}次，放弃。考虑升级到付费套餐或使用其他模型。"
                    ) from exc
                # 指数退避等待
                wait = min(self.rate_limit_base_wait * (2 ** (rate_limit_hits - 1)), 60)
                logger.warning(
                    f"被限流，等待 {wait:.0f}s "
                    f"(限流命中 {rate_limit_hits}/{self.rate_limit_retries})"
                )
                time.sleep(wait)
                last_error = "超过速率限制"

            except litellm.Timeout:
                # 超时错误
                error_retries += 1
                logger.warning(f"超时 (尝试 {error_retries}/{self.max_retries})")
                last_error = "请求超时"

            except KeyboardInterrupt:
                # 用户中断，不重试，直接抛出
                raise
            except Exception as e:
                # 其他异常
                error_retries += 1
                logger.warning(f"LLM调用失败: {e} (尝试 {error_retries}/{self.max_retries})")
                last_error = str(e)
                if error_retries < self.max_retries:
                    # 等待1秒后重试
                    time.sleep(1)

        # 达到最大重试次数仍然失败
        raise RuntimeError(f"LLM调用在{error_retries}次尝试后失败: {last_error}")

    def call_json(self, prompt: str, system_message: str | None = None) -> dict:
        """调用LLM并将响应解析为JSON。"""
        text = self.call(prompt, system_message=system_message)
        return parse_llm_json(text)

    async def acall(self, prompt: str, system_message: str | None = None) -> str:
        """call()的异步版本。通过共享令牌桶进行限速。"""
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
                    logger.warning(f"空响应 (尝试 {error_retries}/{self.max_retries})")
                    if error_retries < self.max_retries:
                        await asyncio.sleep(1)
                        continue
                    raise RuntimeError(f"LLM在{self.max_retries}次尝试后返回空响应")

                return text

            except litellm.RateLimitError as exc:
                rate_limit_hits += 1
                if rate_limit_hits > self.rate_limit_retries:
                    raise RuntimeError(
                        f"被限流{rate_limit_hits}次，放弃。考虑升级到付费套餐或使用其他模型。"
                    ) from exc
                wait = min(self.rate_limit_base_wait * (2 ** (rate_limit_hits - 1)), 60)
                logger.warning(
                    f"被限流，等待 {wait:.0f}s "
                    f"(限流命中 {rate_limit_hits}/{self.rate_limit_retries})"
                )
                await asyncio.sleep(wait)
                last_error = "超过速率限制"

            except asyncio.CancelledError:
                # 异步任务被取消，直接抛出
                raise
            except litellm.Timeout:
                error_retries += 1
                logger.warning(f"超时 (尝试 {error_retries}/{self.max_retries})")
                last_error = "请求超时"

            except KeyboardInterrupt:
                raise
            except Exception as e:
                error_retries += 1
                logger.warning(f"LLM调用失败: {e} (尝试 {error_retries}/{self.max_retries})")
                last_error = str(e)
                if error_retries < self.max_retries:
                    await asyncio.sleep(1)

        raise RuntimeError(f"LLM调用在{error_retries}次尝试后失败: {last_error}")

    async def acall_json(self, prompt: str, system_message: str | None = None) -> dict:
        """call_json()的异步版本。"""
        text = await self.acall(prompt, system_message=system_message)
        return parse_llm_json(text)


def parse_llm_json(text: str) -> dict:
    """从LLM响应解析JSON，处理常见的输出异常。

    LLM经常会把JSON包装在markdown代码块中，或者在JSON前后添加解释文本。
    这个函数会去除这些多余内容并尝试提取有效JSON。
    """
    # 去除markdown代码块标记
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())

    # 首先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 解析失败，尝试通过扫描平衡括号找到有效的JSON对象
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            # 从当前位置开始扫描，找到匹配的闭合括号
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    # 找到平衡的JSON对象，尝试解析
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        break
            break

    # 所有尝试都失败，抛出异常
    raise ValueError(f"无法从LLM响应解析JSON: {text[:200]}...")
