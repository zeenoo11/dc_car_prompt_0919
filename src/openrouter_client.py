import json
import re
import time
from typing import Dict, List, Optional

import requests

from .config import AppConfig


class OpenRouterClient:
    """OpenRouter Chat Completions API 래퍼.

    - 지정된 모델과 시스템 프롬프트, 사용자 입력으로 분류 요청을 전송합니다.
    - 네트워크 오류, 429, 5xx에 대해 제한적 재시도를 수행합니다.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._session = requests.Session()

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 1,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> str:
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.app_title:
            headers["X-Title"] = self.config.app_title

        payload: Dict[str, object] = {
            "model": model or self.config.model_id,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else 1,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._session.post(
                    url, headers=headers, data=json.dumps(payload), timeout=self.config.request_timeout_seconds
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                # 429/5xx는 재시도
                if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(retry_backoff_seconds * attempt)
                    continue
                # 그 외는 즉시 예외
                response.raise_for_status()
            except Exception as exc:  # 네트워크/파싱 오류 등
                last_error = exc
                if attempt < max_retries:
                    time.sleep(retry_backoff_seconds * attempt)
                    continue
                raise
        # 이 지점은 보통 닿지 않음
        if last_error:
            raise last_error
        raise RuntimeError("알 수 없는 오류로 응답을 받지 못했습니다.")

    @staticmethod
    def _normalize_to_label(text: str) -> Optional[int]:
        """모델 응답에서 첫 번째 0/1 숫자를 추출하여 정수로 변환합니다."""
        if not text:
            return None
        match = re.search(r"([01])", text)
        if not match:
            return None
        return int(match.group(1))

    def classify_binary(self, system_prompt: str, title: str, content: str) -> Optional[int]:
        """제목/본문을 입력으로 받아 0 또는 1을 반환하도록 요청합니다.

        시스템 프롬프트에 출력 형식을 강하게 고지하고, 사용자 메시지에서도 반복 고지합니다.
        """
        user_text = (
            f"다음 한국어 뉴스의 자동차 산업 직접 관련성 여부를 0/1로 분류하세요.\n"
            f"제목: {title}\n"
            f"본문: {content}\n\n"
            f"오직 숫자 하나(0 또는 1)만 출력하세요. 설명/여분 텍스트 금지."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\n출력 형식: 오직 숫자 하나('0' 또는 '1')만 출력. 여분 텍스트 절대 금지."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        raw = self.chat(messages)
        return self._normalize_to_label(raw) 