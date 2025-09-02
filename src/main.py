import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from datetime import datetime
import logging
from math import sqrt
from pathlib import Path

try:
    from .config import AppConfig
    from .openrouter_client import OpenRouterClient
except ImportError:
    # 'python src/main.py'로 직접 실행 시 패키지 컨텍스트 폴백
    import sys as _sys
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    from src.config import AppConfig
    from src.openrouter_client import OpenRouterClient


logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    system_prompt: str
    accuracy: float
    total: int
    correct: int
    wrong: int
    length_chars: int
    length_score: float
    composite: float


def load_dataset(csv_path: str) -> pd.DataFrame:
    """CSV를 불러옵니다. 필수 열: ID, title, content, label"""
    df = pd.read_csv(csv_path)
    for col in ["title", "content", "label"]:
        if col not in df.columns:
            raise ValueError(f"CSV에 '{col}' 열이 없습니다.")
    return df


def _length_score(prompt: str, base: int = 3000) -> float:
    """root(1 - (len/3000)^2) 길이 점수. 길이가 base 이상이면 0."""
    n = len(prompt)
    if n >= base:
        return 0.0
    ratio = n / float(base)
    return sqrt(max(0.0, 1.0 - ratio * ratio))


def generate_prompt_candidates() -> List[str]:
    """시스템 프롬프트 후보들을 생성합니다. (README 요구: 시스템 프롬프트 변경 평가)"""
    base = (
        "너는 한국어 뉴스 텍스트를 대상으로 하는 매우 엄격한 이진 분류기다.\n"
        "판단 기준:\n"
        "- 1: 자동차/모빌리티 산업에 '직접적'으로 관련됨. (예: 완성차/OEM, 차량 판매/생산/공장, 모델명, 부품(배터리/모터/차량용 반도체/카메라), 충전/충전기, 자율주행/ADAS/SDV, 차량용 소프트웨어/플랫폼)\n"
        "- 0: 간접·주변·거시적 맥락(관세/무역전쟁/정치, AI/보안 일반, 거시경제, 비자동차 산업 일반) 또는 관련성이 약한 경우\n"
        "경계 사례 처리:\n"
        "- 기업 실적/정책/무역 이슈라도 자동차 생산/판매/모델/부품 등 구체적 차량 맥락이 없으면 0\n"
        "- 전기차/배터리/부품이 자동차 적용을 전제로 구체적으로 논의되면 1\n"
        "- 단순 키워드 언급만 있고 맥락이 비자동차 중심이면 0\n"
    )

    variants = [
        base + "\n분류 임계값: 관련성에 대한 확신이 없으면 0으로 분류하라.",
        base + "\n가중치: '현대차, 기아, BYD, 테슬라, LG이노텍, LG에너지솔루션, 삼성SDI, 포티투닷' 등 브랜드/부품사가 구체적으로 등장하고 차량 맥락이면 1을 우선 고려하라.",
        base + "\n부정 규칙: 관세/정치/거시 경제 이슈가 주제이고 자동차는 사례로만 등장하면 0.",
        base + "\n긍정 규칙: 생산/판매 수치, 공장/라인/양산, 차종/모델/플랫폼/모듈이 핵심이면 1.",
        base + "\n타협 불가: 출력은 반드시 '0' 또는 '1' 중 하나의 숫자만. 애매하면 0.",
        base + "\n정밀 규칙: 차량용 카메라/센서/AP 모듈/EV 충전/EVCC/배터리-ESS가 자동차 용도 중심이면 1, 일반 IT/전력/보안 용도 중심이면 0.",
    ]

    # 추가: 길이 400자 내외를 목표로 하는 초간결형/함정 명시형/예시 1줄형
    micro = (
        "목표: 한국어 뉴스의 '자동차 직접 관련성'을 0/1로 분류.\n"
        "1: 완성차/OEM, 모델/생산·판매·공장/양산, 부품(배터리·모터·차량용 반도체·카메라·EVCC), 충전/충전소, 자율주행/ADAS/SDV/V2X/차량SW가 기사 핵심.\n"
        "0: 관세·정치·거시·보안·AI 일반·ESS(비차량)·스마트폰/일반 전자 등 비차량 중심. 브랜드명만 스치면 0. 애매하면 0.\n"
        "출력: 0 또는 1만."
    )
    traps = (
        "규칙: '브랜드 언급만'=0, '차량 적용 불명확한 배터리/카메라/반도체/ESS'=0, '정책/관세/금리/보안 일반'=0, '생산/판매/공장/양산/모델/부품이 핵심'=1. 출력은 숫자 하나."
    )
    mini_example = (
        "예시 1=1: 'EV 배터리·EVCC 양산 시작', 예시 2=0: '패스키 보안 도입'. 규칙을 따르며 0/1만 출력."
    )
    variants.extend([micro, traps, mini_example])
    return variants


def evaluate_prompt(
    client: OpenRouterClient, df: pd.DataFrame, system_prompt: str, max_workers: int
) -> Tuple[EvalResult, List[Optional[int]]]:
    """주어진 시스템 프롬프트로 전체 데이터셋을 평가합니다."""
    titles: List[str] = df["title"].astype(str).tolist()
    contents: List[str] = df["content"].astype(str).tolist()
    golds: List[int] = df["label"].astype(int).tolist()

    predictions: List[Optional[int]] = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(client.classify_binary, system_prompt, titles[i], contents[i]): i
            for i in range(len(df))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="inference"):
            idx = futures[future]
            try:
                pred = future.result()
            except Exception:
                pred = None
            predictions[idx] = pred

    correct = 0
    total = len(golds)
    for g, p in zip(golds, predictions):
        if p is not None and p == g:
            correct += 1
    accuracy = correct / total if total else 0.0
    lchars = len(system_prompt)
    lscore = _length_score(system_prompt)
    composite = 0.9 * accuracy + 0.1 * lscore
    return (
        EvalResult(
            system_prompt=system_prompt,
            accuracy=accuracy,
            total=total,
            correct=correct,
            wrong=total - correct,
            length_chars=lchars,
            length_score=lscore,
            composite=composite,
        ),
        predictions,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="시스템 프롬프트별 OpenRouter 분류 정확도 평가")
    parser.add_argument("--samples", type=str, default="samples.csv", help="CSV 경로")
    parser.add_argument("--eval-samples", type=str, default=None, help="추가 평가용 CSV (있으면 이 데이터로 점수 산출)")
    parser.add_argument("--target-acc", type=float, default=0.9, help="목표 정확도")
    parser.add_argument("--max-prompts", type=int, default=12, help="최대 평가 프롬프트 수")
    parser.add_argument("--max-workers", type=int, default=None, help="동시 요청 스레드 수")
    args = parser.parse_args()

    # 설정 로드
    config = AppConfig.from_env()
    if args.max_workers is not None:
        config.max_workers = args.max_workers
    config.dataset_path = args.samples
    config.target_accuracy = args.target_acc
    config.max_trials = args.max_prompts

    # 데이터 로드
    df_train = load_dataset(config.dataset_path)
    df_eval = load_dataset(args.eval_samples) if args.eval_samples else None
    dataset_for_scoring = df_eval if df_eval is not None else df_train

    # 클라이언트 준비
    client = OpenRouterClient(config)

    # 프롬프트 후보 준비
    candidates = generate_prompt_candidates()
    if config.max_trials:
        candidates = candidates[: config.max_trials]

    # 결과 폴더 생성
    os.makedirs("results", exist_ok=True)

    # 식별자
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_name = Path(config.dataset_path).stem
    eval_name = Path(args.eval_samples).stem if args.eval_samples else train_name

    # 평가 루프
    best: Optional[EvalResult] = None
    all_results: List[EvalResult] = []
    best_predictions: Optional[List[Optional[int]]] = None
    for idx, prompt in enumerate(candidates, start=1):
        logger.info("[%d/%d] 프롬프트 평가 중... len=%d", idx, len(candidates), len(prompt))
        result, preds = evaluate_prompt(client, dataset_for_scoring, prompt, config.max_workers)
        all_results.append(result)
        logger.info("정확도=%.4f, 길이점수=%.4f, 종합=%.4f (정답 %d/%d)", result.accuracy, result.length_score, result.composite, result.correct, result.total)
        if best is None or result.composite > best.composite:
            best = result
            best_predictions = preds
        if best and best.accuracy >= config.target_accuracy:
            logger.info("목표 정확도 달성: %.4f >= %.4f", best.accuracy, config.target_accuracy)
            # 계속 탐색할지 여부는 필요 시 옵션화
            # break

    # 결과 저장 (고유 파일명)
    summary = {
        "model": config.model_id,
        "temperature": config.temperature,
        "target_accuracy": config.target_accuracy,
        "best": {
            "accuracy": best.accuracy if best else 0.0,
            "length_chars": best.length_chars if best else 0,
            "length_score": best.length_score if best else 0.0,
            "composite": best.composite if best else 0.0,
            "context" : best.system_prompt if best else "",
        },
        "trials": len(all_results),
        "results": [
            {
                "accuracy": r.accuracy,
                "length_chars": r.length_chars,
                "length_score": r.length_score,
                "composite": r.composite,
                "correct": r.correct,
                "wrong": r.wrong,
                "context" : r.system_prompt,
            }
            for r in all_results
        ],
    }
    metrics_path = os.path.join("results", f"metrics_{eval_name}_{ts}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("메트릭 저장: %s", metrics_path)

    if best:
        bp_path = os.path.join("results", f"best_prompt_{eval_name}_{ts}.txt")
        with open(bp_path, "w", encoding="utf-8") as f:
            f.write(best.system_prompt)
        logger.info("최적 프롬프트 저장: %s", bp_path)

    # 예측 CSV 저장 (평가 데이터 기준)
    if best and best_predictions is not None:
        out_df = pd.DataFrame(
            {
                "pred": pd.Series(best_predictions, dtype="Int64"),
                "label": dataset_for_scoring["label"].astype("Int64"),
            }
        )
        if "ID" in dataset_for_scoring.columns:
            out_df.insert(0, "ID", dataset_for_scoring["ID"])
        if "title" in dataset_for_scoring.columns:
            out_df.insert(1 if "ID" in dataset_for_scoring.columns else 0, "title", dataset_for_scoring["title"])
        pred_path = os.path.join("results", f"prediction_{eval_name}_{ts}.csv")
        out_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info("예측 CSV 저장: %s", pred_path)

    # 콘솔 요약
    if best:
        print("\n최적 프롬프트 종합 점수:", f"{best.composite:.4f}")
        print("최적 프롬프트 정확도:", f"{best.accuracy:.4f}")
        print("최적 프롬프트 길이점수:", f"{best.length_score:.4f}")
    else:
        print("\n유효한 결과가 생성되지 않았습니다.")


if __name__ == "__main__":
    main() 
    # uv run src/main.py --samples samples.csv --eval-samples samples2.csv --target-acc 0.9 --max-prompts 12 --max-workers 10