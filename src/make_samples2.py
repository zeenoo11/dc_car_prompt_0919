import argparse
import csv
import os
import random
import logging
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    from .crawl_rss import collect_from_feeds
    from .augment_llm import paraphrase_batch
    from .config import AppConfig
    from .openrouter_client import OpenRouterClient
except ImportError:
    # 직접 실행 폴백
    import sys as _sys, os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    from src.crawl_rss import collect_from_feeds
    from src.augment_llm import paraphrase_batch
    from src.config import AppConfig
    from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

CAR_FEEDS = [
    # 국제 일반(예시)
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=autos-transport&post_type=best",
    "https://www.reuters.com/markets/companies/autos-transportation/rss/",
    "https://rss.nytimes.com/services/xml/rss/nyt/Automobiles.xml",
    # 국내/기술 섞기(예시, 가용성은 매체 정책에 따라 다를 수 있음)
    "https://www.hankyung.com/feed/it",
    "https://www.hankyung.com/feed/economy",
]

NEG_FEEDS = [
    # 비자동차/거시/정치/보안 등
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.reuters.com/world/rss/",
    "https://www.bleepingcomputer.com/feed/",  # 보안
    "https://feeds.bbci.co.uk/news/business/rss.xml",
]


def label_by_rules(title: str, content: str) -> int:
    """간단한 규칙 라벨링(초안). 수동 검수 전 하드 라벨."""
    text = (title or "") + "\n" + (content or "")
    positives = [
        "현대차", "기아", "BYD", "Tesla", "테슬라", "아이오닉", "코나", "EV3", "전기차",
        "자율주행", "ADAS", "SDV", "V2X", "로보택시",
        "배터리", "모터", "차량용", "반도체", "카메라", "EVCC", "충전기", "충전소",
        "공장", "양산", "생산", "판매",
    ]
    negatives_hint = [
        "관세", "정치", "대선", "무역", "금리", "인플레이션", "패스키", "보안", "해킹",
        "로봇", "로보틱스", "에너지 저장", "ESS", "스마트폰", "모바일 카메라",
    ]
    pos = any(k in text for k in positives)
    neg = any(k in text for k in negatives_hint)
    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    # 경계: 차량 적용 명시가 있으면 1 우선
    boundary = ["차량용", "차량", "모빌리티", "OEM", "모델", "라인", "양산"]
    if any(k in text for k in boundary):
        return 1
    return 0

# 합성 데이터 생성용 자원
OEM_BRANDS = ["현대차", "기아", "테슬라", "BYD", "도요타", "GM", "포드", "폭스바겐"]
COMP_BRANDS = ["LG이노텍", "LG에너지솔루션", "삼성SDI", "보쉬", "퀄컴", "NXP"]
CITIES = ["서울", "부산", "울산", "화성", "테슬라 프리몬트", "오스틴", "상하이", "멕시코시티"]
COUNTRIES = ["한국", "미국", "중국", "일본", "독일", "멕시코", "헝가리", "폴란드"]
NUMS = list(range(3, 101))


def synth_positive_article() -> Tuple[str, str]:
    brand = random.choice(OEM_BRANDS + COMP_BRANDS)
    city = random.choice(CITIES)
    country = random.choice(COUNTRIES)
    qty = random.choice(NUMS)
    topic = random.choice([
        "전기차 생산라인 증설", "양산 시작", "EV 충전소 구축", "EVCC 양산", "차량용 AP 모듈 양산",
        "자율주행 시범 운행", "ADAS 플랫폼 고도화", "SDV 소프트웨어 업데이트",
    ])
    title = f"{brand} {country} {city}에서 {topic} 발표"
    paras = [
        f"{brand}가 {country} {city}에서 {topic} 계획을 공식화했다.",
        f"이번 계획은 연간 {qty}만 대 규모 생산/공급을 목표로 하며, 공장/라인/양산 지표가 개선될 전망이다.",
        f"프로그램에는 배터리·모터·차량용 반도체·카메라·EVCC 등 핵심 부품 공급망 강화가 포함된다.",
        "자율주행/ADAS/SDV/V2X 등 차량 소프트웨어 역량도 동시에 고도화한다.",
    ]
    content = "\n".join(paras) * 2
    return title, content


def synth_negative_article() -> Tuple[str, str]:
    topic = random.choice([
        "관세 정책 논쟁", "대선 공약 발표", "금리 인상 전망", "패스키 보안 도입", "주택용 ESS 확대",
        "스마트폰 카메라 모듈 출시", "클라우드 보안 사고", "로봇 자동화 일반",
    ])
    region = random.choice(["미국", "EU", "중국", "한국", "일본"])
    title = f"{region} {topic} 확산"
    paras = [
        f"{region}에서 {topic} 이슈가 부각되고 있다.",
        "정책/보안/에너지/전자 기기 전반의 동향으로, 자동차 '직접' 맥락은 핵심이 아니다.",
        "거시경제 변수와 규제 변화가 시장에 영향을 줄 전망이다.",
    ]
    content = "\n".join(paras) * 3
    return title, content


def synth_borderline_article() -> Tuple[str, str]:
    brand = random.choice(OEM_BRANDS + COMP_BRANDS)
    theme = random.choice(["기술 협력 논의", "공동 연구 MOU", "친환경 경영 보고서", "해외 전시회 참가"])
    title = f"{brand} {theme}"
    paras = [
        f"{brand}가 {theme}를 진행했다.",
        "기사 전반은 기업 일반/브랜드 활동 중심이며 차량의 생산/판매/부품/충전/자율주행 맥락은 제한적이다.",
        "추후 구체적 차량 적용이 언급되기 전까지는 직접 관련성 판단이 어렵다.",
    ]
    content = "\n".join(paras) * 3
    return title, content


def augment_synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for i in range(n):
        mode = random.random()
        if mode < 0.45:
            t, c = synth_positive_article()
            y = 1
        elif mode < 0.8:
            t, c = synth_negative_article()
            y = 0
        else:
            t, c = synth_borderline_article()
            y = 0
        rows.append({"ID": f"SYN_{i:04d}", "title": t, "content": c, "label": y})
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    df["content_len"] = df["content"].apply(lambda x: len(x or ""))
    df = df[(df["content_len"] >= 200) & (df["content_len"] <= 6000)].drop(columns=["content_len"]).reset_index(drop=True)
    return df


def build_samples2(target_count: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    logger.info("build_samples2 start target=%d", target_count)

    pos_rows: List[Tuple[str, str, str, str]] = collect_from_feeds(CAR_FEEDS, max_items_per_feed=500)
    neg_rows: List[Tuple[str, str, str, str]] = collect_from_feeds(NEG_FEEDS, max_items_per_feed=500)

    rows: List[Tuple[str, str]] = []
    for src, link, title, content in pos_rows:
        rows.append((title, content))
        if len(rows) >= target_count // 2:
            break
    for src, link, title, content in neg_rows:
        rows.append((title, content))
        if len(rows) >= target_count:
            break

    all_rows = pos_rows + neg_rows
    i = 0
    while len(rows) < target_count and i < len(all_rows):
        _, _, title, content = all_rows[i]
        rows.append((title, content))
        i += 1

    labeled = []
    for idx, (title, content) in enumerate(rows):
        y = label_by_rules(title, content)
        labeled.append({"ID": f"S2_{idx:04d}", "title": title, "content": content, "label": y})

    df = pd.DataFrame(labeled)
    logger.info("initial collected rows=%d", len(df))

    if len(df) < target_count:
        need = target_count - len(df)
        logger.info("synthetic augment need=%d", need)
        syn = augment_synthetic(need, seed=seed)
        df = pd.concat([df, syn], ignore_index=True)

    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    df["content_len"] = df["content"].apply(lambda x: len(x or ""))
    df = df[(df["content_len"] >= 200) & (df["content_len"] <= 8000)].drop(columns=["content_len"]).reset_index(drop=True)

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info("after filter pos=%d neg=%d total=%d", n_pos, n_neg, len(df))
    if min(n_pos, n_neg) > 0:
        n_min = min(n_pos, n_neg)
        pos_df = df[df["label"] == 1].sample(n=n_min, random_state=seed)
        neg_df = df[df["label"] == 0].sample(n=n_min, random_state=seed)
        df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        logger.info("balanced to %d per class", n_min)

    if len(df) > target_count:
        df = df.head(target_count)

    df = df.reset_index(drop=True)
    df["ID"] = [f"S2_{i:04d}" for i in range(len(df))]

    logger.info("build_samples2 done total=%d", len(df))
    return df


def _paraphrase_df(df: pd.DataFrame, frac: float, max_workers: int) -> pd.DataFrame:
    frac = max(0.0, min(1.0, frac))
    if frac == 0 or len(df) == 0:
        return df
    k = max(1, int(len(df) * frac))
    target_idx = df.sample(n=k, random_state=42).index.tolist()
    logger.info("paraphrase target %d rows (%.2f)", k, frac)

    cfg = AppConfig.from_env()
    client = OpenRouterClient(cfg)

    rows = [(i, df.at[i, "title"], df.at[i, "content"], int(df.at[i, "label"])) for i in target_idx]
    outs = paraphrase_batch(client, rows, max_workers=max_workers, temperature=0.7, max_tokens=380)
    for i, new_t, new_c in outs:
        df.at[i, "title"] = new_t
        df.at[i, "content"] = new_c
    return df


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="samples2 생성 스크립트(RSS+규칙 라벨링+합성 보충+LLM 패러프레이즈)")
    parser.add_argument("--out", type=str, default="samples2.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--paraphrase-frac", type=float, default=0.0, help="LLM 패러프레이즈 비율(0~1)")
    parser.add_argument("--paraphrase-max-workers", type=int, default=4, help="패러프레이즈 동시 처리 수")
    args = parser.parse_args()

    df = build_samples2(target_count=args.n)

    if args.paraphrase_frac > 0:
        try:
            df = _paraphrase_df(df, frac=args.paraphrase_frac, max_workers=args.paraphrase_max_workers)
            logger.info("paraphrase completed")
        except Exception as exc:
            logger.warning("paraphrase skipped due to error: %s", exc)

    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    logger.info("saved: %s (%d rows)", args.out, len(df))


if __name__ == "__main__":
    main() 
    # uv run src/make_samples2.py --out samples2.csv --n 10000 --paraphrase-frac 0.1 --paraphrase-max-workers 4