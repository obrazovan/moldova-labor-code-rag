"""
Скрипт генерации наглядных графиков и диаграмм для отчета и README.md (Этап 6).
Создает графики высокого разрешения (300 DPI) в папке experiments/charts/:
1. exp1_top_k_metrics.png — Динамика Hit Rate, Recall, Precision и MRR в зависимости от K.
2. exp1_category_comparison.png — Сравнение метрик по 4 категориям запросов.
3. exp3_embeddings_comparison.png — Сравнение rubert-tiny2 vs multilingual-e5-small.
4. exp4_chunking_comparison.png — Сравнение 4 стратегий чанкинга.
5. exp2_reranker_comparison.png — Сравнение плотного поиска (Dense) и FlashRank.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Настройка единого современного стиля оформления
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#D0D7DE"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#E1E4E8"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

CHARTS_DIR = Path(__file__).resolve().parent / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = Path(__file__).resolve().parent / "benchmark_results.json"


def load_benchmark_data() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Файл {RESULTS_PATH} не найден. Сначала выполните run_experiments.py")
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_exp1_top_k(data: dict) -> None:
    """График 1: Влияние Top-K на Hit Rate, Recall, Precision и MRR."""
    exp1 = data["experiment_1_top_k"]["overall"]
    k_vals = data["experiment_1_top_k"].get("k_values", [1, 2, 3, 5, 10, 20])
    hit_rates = [exp1[f"hit_rate@{k}"] * 100 for k in k_vals]
    recalls = [exp1[f"recall@{k}"] * 100 for k in k_vals]
    precisions = [exp1[f"precision@{k}"] * 100 for k in k_vals]
    mrrs = [exp1.get(f"mrr@{k}", exp1["mrr"]) * 100 for k in k_vals]

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)

    x = np.arange(len(k_vals))
    width = 0.19

    rects1 = ax.bar(x - 1.5 * width, hit_rates, width, label="Hit Rate@K (%)", color="#1F77B4", edgecolor="none")
    rects2 = ax.bar(x - 0.5 * width, recalls, width, label="Recall@K (%)", color="#2CA02C", edgecolor="none")
    rects3 = ax.bar(x + 0.5 * width, mrrs, width, label="MRR@K (%)", color="#FF7F0E", edgecolor="none")
    rects4 = ax.bar(x + 1.5 * width, precisions, width, label="Precision@K (%)", color="#9467BD", edgecolor="none")

    # Добавление значений над столбцами
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    ax.set_ylabel("Значение метрики (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Эксперимент 1: Метрики поиска при K in [1, 2, 3, 5, 10, 20] (multilingual-e5-small)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in k_vals], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    # Пояснительная аннотация об эффекте насыщения
    ax.text(
        0.02,
        0.03,
        "• Hit Rate@K и MRR@K круто растут от K=1 (80.8%) до K=3 (96.2%), затем выходят на плато,\n"
        "  поскольку для 25 из 26 запросов первая целевая статья находится уже в Top-3 (ранг ≤ 3).\n"
        "• Recall@K монотонно растет с 69.2% до 96.2% за счет захвата 2-й и 3-й статей для multi_context.\n"
        "• Precision@K закономерно снижается от 80.8% до 6.0% по мере расширения пула кандидатов.",
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.9),
    )

    plt.tight_layout()
    output_file = CHARTS_DIR / "exp1_top_k_metrics.png"
    plt.savefig(output_file)
    plt.close()
    print(f"График сохранен: {output_file}")


def plot_exp1_categories(data: dict) -> None:
    """График 2: Метрики качества в разрезе категорий запросов."""
    cats = data["experiment_1_top_k"]["by_category"]
    cat_names = ["factual", "multi_context", "specific_search", "synonym_slang"]
    cat_labels = ["Фактологические\n(factual, 8)", "Многоконтекстные\n(multi_context, 4)", "Профильные нормы\n(specific, 8)", "Разговорные / сленг\n(synonym_slang, 6)"]

    hr3 = [cats[c]["hit_rate@3"] * 100 for c in cat_names]
    mrr = [cats[c]["mrr"] * 100 for c in cat_names]
    rec5 = [cats[c]["recall@5"] * 100 for c in cat_names]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    x = np.arange(len(cat_names))
    width = 0.25

    r1 = ax.bar(x - width, hr3, width, label="Hit Rate@3 (%)", color="#2563EB", edgecolor="none")
    r2 = ax.bar(x, mrr, width, label="MRR (%)", color="#F59E0B", edgecolor="none")
    r3 = ax.bar(x + width, rec5, width, label="Recall@5 (%)", color="#10B981", edgecolor="none")

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_ylabel("Процент (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Эксперимент 1: Сравнение качества поиска по 4 категориям запросов", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    plt.tight_layout()
    output_file = CHARTS_DIR / "exp1_category_comparison.png"
    plt.savefig(output_file)
    plt.close()
    print(f"График сохранен: {output_file}")


def plot_exp3_embeddings(data: dict) -> None:
    """График 3: Сравнение моделей эмбеддингов (rubert-tiny2 vs multilingual-e5-small)."""
    exp3 = data["experiment_3_embeddings"]
    rubert = exp3["rubert_tiny2"]
    e5 = exp3["multilingual_e5"]

    metrics = ["Hit Rate@3", "Hit Rate@5", "MRR", "Latency (мс)"]
    rubert_vals = [rubert["hit_rate@3"] * 100, rubert["hit_rate@5"] * 100, rubert["mrr"] * 100, rubert["avg_latency_ms"]]
    e5_vals = [e5["hit_rate@3"] * 100, e5["hit_rate@5"] * 100, e5["mrr"] * 100, e5["avg_latency_ms"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300, gridspec_kw={"width_ratios": [2.5, 1]})

    x = np.arange(3)
    width = 0.32

    # Метрики качества (проценты)
    b1 = ax1.bar(x - width / 2, rubert_vals[:3], width, label="rubert-tiny2 (29M)", color="#64748B", edgecolor="none")
    b2 = ax1.bar(x + width / 2, e5_vals[:3], width, label="multilingual-e5-small (118M)", color="#0284C7", edgecolor="none")

    for rects in [b1, b2]:
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Показатель (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Сравнение точности поиска (IR Metrics)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Hit Rate@3", "Hit Rate@5", "MRR"], fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # Метрика скорости (задержка в мс)
    x_lat = np.array([0, 1])
    lat_bars = ax2.bar(x_lat, [rubert_vals[3], e5_vals[3]], width=0.5, color=["#64748B", "#0284C7"], edgecolor="none")
    for rect in lat_bars:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f} ms", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Средняя задержка (мс)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Время ответа (Latency)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_lat)
    ax2.set_xticklabels(["rubert-tiny2", "e5-small"], fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 45)

    plt.suptitle("Эксперимент 3: Сравнение моделей эмбеддингов", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_file = CHARTS_DIR / "exp3_embeddings_comparison.png"
    plt.savefig(output_file)
    plt.close()
    print(f"График сохранен: {output_file}")


def plot_exp4_chunking(data: dict) -> None:
    """График 4: Сравнение 4 стратегий чанкинга."""
    exp4 = data["experiment_4_chunking"]["strategies"]
    strats = ["by_article", "fixed_size", "fixed_overlap", "by_paragraph"]
    labels = ["by_article\n(по статьям)", "fixed_size\n(500 симв.)", "fixed_overlap\n(500/100 симв.)", "by_paragraph\n(по абзацам)"]

    hr3 = [exp4[s]["hit_rate@3"] * 100 for s in strats]
    hr5 = [exp4[s]["hit_rate@5"] * 100 for s in strats]
    mrr = [exp4[s]["mrr"] * 100 for s in strats]
    chunks = [exp4[s]["total_chunks"] for s in strats]

    fig, ax1 = plt.subplots(figsize=(10, 5.2), dpi=300)

    x = np.arange(len(strats))
    width = 0.24

    r1 = ax1.bar(x - width, hr3, width, label="Hit Rate@3 (%)", color="#3B82F6", edgecolor="none")
    r2 = ax1.bar(x, hr5, width, label="Hit Rate@5 (%)", color="#10B981", edgecolor="none")
    r3 = ax1.bar(x + width, mrr, width, label="MRR (%)", color="#F59E0B", edgecolor="none")

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_ylabel("Показатель (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_title("Эксперимент 4: Сравнение стратегий чанкинга (на модели rubert-tiny2)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 85)
    ax1.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    # Добавляем текстовые плашки с количеством чанков
    for idx, count in enumerate(chunks):
        ax1.text(
            idx,
            75,
            f"Индекс:\n{count} чанков",
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.9),
        )

    plt.tight_layout()
    output_file = CHARTS_DIR / "exp4_chunking_comparison.png"
    plt.savefig(output_file)
    plt.close()
    print(f"График сохранен: {output_file}")


def plot_exp2_reranker(data: dict) -> None:
    """График 5: Сравнение Dense Only vs FlashRank."""
    exp2 = data["experiment_2_reranker"]
    dense = exp2["dense"]
    reranked = exp2["reranked"]

    metrics = ["Hit Rate@3", "Hit Rate@5", "Hit Rate@10", "MRR"]
    dense_vals = [dense["hit_rate@3"] * 100, dense["hit_rate@5"] * 100, dense["hit_rate@10"] * 100, dense["mrr"] * 100]
    rerank_vals = [reranked["hit_rate@3"] * 100, reranked["hit_rate@5"] * 100, reranked["hit_rate@10"] * 100, reranked["mrr"] * 100]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)

    x = np.arange(len(metrics))
    width = 0.35

    r1 = ax.bar(x - width / 2, dense_vals, width, label="Dense Only (multilingual-e5-small, 30.5 ms)", color="#059669", edgecolor="none")
    r2 = ax.bar(x + width / 2, rerank_vals, width, label="Dense + FlashRank (ms-marco-MultiBERT, 2501 ms)", color="#DC2626", edgecolor="none")

    for rects in [r1, r2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Показатель (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Эксперимент 2: Влияние реранкера (Dense Only vs FlashRank)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    plt.tight_layout()
    output_file = CHARTS_DIR / "exp2_reranker_comparison.png"
    plt.savefig(output_file)
    plt.close()
    print(f"График сохранен: {output_file}")


def main() -> None:
    data = load_benchmark_data()
    print("Генерация диаграмм на основе benchmark_results.json...")
    plot_exp1_top_k(data)
    plot_exp1_categories(data)
    plot_exp3_embeddings(data)
    plot_exp4_chunking(data)
    plot_exp2_reranker(data)
    print("Все 5 графиков успешно сгенерированы в папке experiments/charts/")


if __name__ == "__main__":
    main()
