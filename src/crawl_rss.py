import re
import time
import logging
from typing import Iterable, List, Tuple, Optional

import requests
import feedparser
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)

USER_AGENT = "dc_car_prompt_0919/0.1 (+https://example.com)"
TIMEOUT_SECONDS = 15


def fetch_url_text(url: str) -> str:
    """URL의 본문 텍스트를 최대한 단순하게 추출합니다."""
    try:
        logger.debug("fetching url: %s", url)
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("fetch failed: %s (%s)", url, exc)
        return ""
    soup = BeautifulSoup(resp.text, "lxml")
    # 기본 기사 텍스트 추출(단순화)
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = "\n".join(t.strip() for t in soup.stripped_strings)
    logger.debug("extracted length=%d from %s", len(text), url)
    return text[:20000]  # 과도한 길이 제한


def parse_rss(feed_url: str) -> List[Tuple[str, str, str]]:
    """RSS 피드를 파싱해 (title, link, summary) 리스트를 반환합니다."""
    logger.info("parse feed: %s", feed_url)
    d = feedparser.parse(feed_url)
    items: List[Tuple[str, str, str]] = []
    entries = getattr(d, "entries", [])
    logger.info("feed entries: %d (%s)", len(entries), feed_url)
    for entry in entries:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        summary = getattr(entry, "summary", "").strip() if hasattr(entry, "summary") else ""
        if title and link:
            items.append((title, link, summary))
    return items


CAR_KEYWORDS = [
    # 자동차 직접 관련 키워드(양성 후보 필터에 사용)
    "현대차", "기아", "BYD", "테슬라", "아이오닉", "코나", "EV3", "EV", "전기차",
    "자율주행", "ADAS", "SDV", "V2X", "로보택시",
    "배터리", "모터", "차량용", "반도체", "카메라", "AP 모듈", "EVCC", "충전기", "충전소",
    "공장", "양산", "생산", "판매",
]


def is_potential_car(title: str, summary: str) -> bool:
    t = (title or "") + " " + (summary or "")
    return any(k in t for k in CAR_KEYWORDS)


def collect_from_feeds(feed_urls: Iterable[str], max_items_per_feed: int = 100) -> List[Tuple[str, str, str, str]]:
    """RSS 피드 다건 수집.

    반환: (source, link, title, content)
    """
    results: List[Tuple[str, str, str, str]] = []
    for feed in feed_urls:
        logger.info("collect from feed: %s", feed)
        entries = parse_rss(feed)
        count = 0
        for title, link, summary in entries:
            if count >= max_items_per_feed:
                break
            # 자동차 가능성 우선 수집
            if not is_potential_car(title, summary):
                continue
            content = fetch_url_text(link)
            if not content:
                continue
            results.append((feed, link, title, content))
            count += 1
            if count % 10 == 0:
                logger.info("feed %s collected %d items", feed, count)
            time.sleep(0.5)
        logger.info("feed %s done: %d items", feed, count)
    logger.info("total collected: %d", len(results))
    return results 