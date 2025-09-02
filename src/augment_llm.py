import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .openrouter_client import OpenRouterClient


logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[Dict[str, str]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # ```json fenced or trailing text에서 JSON만 추출
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _build_messages(label: int, title: str, content: str):
    sys = (
        "너는 한국어 편집자다. 입력의 의미(사실/맥락)를 유지하면서 제목과 본문을 한국어로 자연스럽게 패러프레이즈하라.\n"
        "- 길이는 원문의 80~120% 범위로 유지\n"
        "- 새로운 사실을 추가 금지, 수치/지명/브랜드 왜곡 금지\n"
        "- 출력은 JSON 한 개만: {\"title\": str, \"content\": str}\n"
        "- 레이블 보존: label=1이면 '자동차 산업 직접 관련' 맥락을 유지, label=0이면 자동차 '직접' 맥락이 되지 않도록 유지\n"
    )
    usr = (
        f"label={label}\n"
        f"title={title}\n"
        f"content={content}\n\n"
        "JSON만 출력"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def paraphrase_batch(
    client: OpenRouterClient,
    rows: List[Tuple[int, str, str, int]],
    max_workers: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 380,
) -> List[Tuple[int, str, str]]:
    """여러 행을 병렬 패러프레이즈. 입력: (idx, title, content, label). 반환: (idx, new_title, new_content)."""
    outputs: List[Tuple[int, str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {}
        for idx, title, content, label in rows:
            messages = _build_messages(label, title, content)
            fut[ex.submit(client.chat, messages, temperature=temperature, max_tokens=max_tokens)] = idx
        for f in as_completed(fut):
            i = fut[f]
            try:
                resp = f.result()
                data = _extract_json(resp)
                if not data or "title" not in data or "content" not in data:
                    logger.warning("paraphrase parse failed at idx=%d, keep original", i)
                    outputs.append((i, rows[i][1], rows[i][2]))
                else:
                    outputs.append((i, str(data["title"]), str(data["content"])) )
            except Exception as exc:
                logger.warning("paraphrase error idx=%d: %s", i, exc)
                outputs.append((i, rows[i][1], rows[i][2]))
    outputs.sort(key=lambda x: x[0])
    return outputs 