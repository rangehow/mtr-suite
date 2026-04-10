"""
OpenAI-compatible API backend for remote sglang/vllm endpoints.

Provides async batch generation with round-robin load balancing across
multiple endpoints. Drop-in replacement for local vllm/sglang backends
when GPUs are on separate machines.

Usage:
    from shared.api_utils import ApiBatchGenerator

    endpoints = [
        "http://<node1-ip>:8080/v1/chat/completions",
        "http://<node2-ip>:8080/v1/chat/completions",
        ...
    ]
    gen = ApiBatchGenerator(
        endpoints=endpoints,
        model="qwen35-fp8",
        max_concurrent=80,           # total in-flight requests
        enable_thinking=False,        # disable reasoning for Qwen3.5
    )
    results = gen.generate(messages_list, sampling_params)
    # results: list[dict] with keys "content", "reasoning_content", "completion_tokens"
"""

import asyncio
import itertools
import time
from typing import Any

import aiohttp
from loguru import logger
from tqdm import tqdm


class ApiBatchGenerator:
    """Async batch generator with round-robin load balancing."""

    def __init__(
        self,
        endpoints: list[str],
        model: str = "qwen35-fp8",
        max_concurrent: int = 80,
        max_retries: int = 5,
        timeout: int = 600,
        enable_thinking: bool = False,
    ):
        self.endpoints = endpoints
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        # Round-robin iterator (thread-safe under asyncio single-thread)
        self._endpoint_cycle = itertools.cycle(endpoints)

    def _next_endpoint(self) -> str:
        return next(self._endpoint_cycle)

    # ---- public sync interface (blocks until all done) ----

    def generate(
        self,
        messages_list: list[list[dict]],
        sampling_params: dict | None = None,
        desc: str | None = None,
    ) -> list[dict]:
        """
        Generate completions for a list of message sequences.

        Args:
            messages_list: list of OpenAI-format message lists.
            sampling_params: dict with temperature, top_p, max_tokens, etc.

        Returns:
            Ordered list of result dicts:
              { "content": str, "reasoning_content": str|None, "completion_tokens": int }
        """
        if sampling_params is None:
            sampling_params = {}
        return asyncio.get_event_loop().run_until_complete(
            self._generate_all(messages_list, sampling_params, desc=desc)
        ) if asyncio.get_event_loop().is_running() else asyncio.run(
            self._generate_all(messages_list, sampling_params, desc=desc)
        )

    def generate_sync(
        self,
        messages_list: list[list[dict]],
        sampling_params: dict | None = None,
        desc: str | None = None,
    ) -> list[dict]:
        """Always create a new event loop (safe from any context)."""
        if sampling_params is None:
            sampling_params = {}
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._generate_all(messages_list, sampling_params, desc=desc)
            )
        finally:
            loop.close()

    # ---- async internals ----

    async def _generate_all(
        self,
        messages_list: list[list[dict]],
        sampling_params: dict,
        desc: str | None = None,
    ) -> list[dict]:
        sem = asyncio.Semaphore(self.max_concurrent)
        results: list[dict | None] = [None] * len(messages_list)
        pbar = tqdm(total=len(messages_list), desc=desc or "API generation",
                   dynamic_ncols=True, mininterval=1.0)

        connector = aiohttp.TCPConnector(limit=self.max_concurrent + 20)
        timeout_obj = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout_obj
        ) as session:
            tasks = []
            for idx, messages in enumerate(messages_list):
                t = asyncio.create_task(
                    self._generate_one(session, sem, idx, messages, sampling_params, results, pbar)
                )
                tasks.append(t)
            await asyncio.gather(*tasks)

        pbar.close()

        # Check for failures
        failed = [i for i, r in enumerate(results) if r is None]
        if failed:
            logger.error(f"{len(failed)} requests failed after all retries: indices {failed[:10]}...")
        return results

    async def _generate_one(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        idx: int,
        messages: list[dict],
        sampling_params: dict,
        results: list,
        pbar,
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": sampling_params.get("temperature", 0.8),
            "top_p": sampling_params.get("top_p", 0.95),
            "max_tokens": sampling_params.get("max_new_tokens",
                          sampling_params.get("max_tokens", 8192)),
        }
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        for attempt in range(1, self.max_retries + 1):
            endpoint = self._next_endpoint()
            try:
                async with sem:
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            logger.warning(
                                f"[{idx}] HTTP {resp.status} from {endpoint} "
                                f"(attempt {attempt}): {body[:200]}"
                            )
                            await asyncio.sleep(min(2 ** attempt, 30))
                            continue
                        data = await resp.json()

                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content")
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)

                results[idx] = {
                    "content": content,
                    "reasoning_content": reasoning,
                    "completion_tokens": completion_tokens,
                }
                pbar.update(1)
                return

            except asyncio.TimeoutError:
                logger.warning(f"[{idx}] Timeout on {endpoint} (attempt {attempt})")
                await asyncio.sleep(min(2 ** attempt, 30))
            except Exception as e:
                logger.warning(f"[{idx}] Error on {endpoint} (attempt {attempt}): {e}")
                await asyncio.sleep(min(2 ** attempt, 30))

        logger.error(f"[{idx}] All {self.max_retries} attempts failed")
        pbar.update(1)
