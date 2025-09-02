import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AppConfig:
    """애플리케이션 전역 설정. 환경 변수에서 불러옵니다."""

    # OpenRouter
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model_id: str = "openai/gpt-4o-mini"
    temperature: float = 0.4
    request_timeout_seconds: int = 60

    # 실행 설정
    max_workers: int = 4
    dataset_path: str = "samples.csv"
    target_accuracy: float = 0.9
    max_trials: int = 12

    # 선택 헤더 (리더보드/식별용)
    http_referer: Optional[str] = None
    app_title: Optional[str] = None

    @staticmethod
    def from_env() -> "AppConfig":
        """환경 변수에서 설정값을 읽어 AppConfig를 반환합니다."""
        # .env 자동 로딩
        load_dotenv()

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY가 설정되지 않았습니다. .env 파일에 키를 설정하세요."
            )

        model_id = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
        temperature_str = os.getenv("OPENROUTER_TEMPERATURE", "0.4").strip()
        temperature = float(temperature_str) if temperature_str else 0.4

        http_referer = os.getenv("APP_HTTP_REFERER")
        app_title = os.getenv("APP_TITLE")

        return AppConfig(
            api_key=api_key,
            model_id=model_id,
            temperature=temperature,
            http_referer=http_referer,
            app_title=app_title,
        ) 