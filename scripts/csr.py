from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from collections import Counter


@dataclass(frozen=True)
class CSRResult:
    """Результаты расчёта CSR-кривой."""
    scaffold_fraction: List[float]
    compound_fraction: List[float]
    auc: float
    f50: Optional[float]
    total_compounds: int
    total_scaffolds: int


def calculate_csr_curve(
    df: pd.DataFrame,
    scaffold_column: str = "murcko_scaffold",
    count_empty_as_no_scaffold: bool = True,
) -> CSRResult:
    """
    Рассчитывает CSR-кривую (Cumulative Scaffold Recovery) для оценки разнообразия скаффолдов.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица с молекулами.
    scaffold_column : str, default 'murcko_scaffold'
        Имя колонки со скаффолдами (SMILES/канонический ключ скаффолда).
    count_empty_as_no_scaffold : bool, default True
        Если True, в знаменателе по соединениям учитываются все строки,
        включая молекулы без скаффолда (пустые строки/NaN). Это делает F50 интерпретируемым
        как доля скаффолдов, покрывающая 50% всего набора, а не только молекул со скаффолдами.

    Returns
    -------
    CSRResult
        Структура с долями по осям, AUC и F50.
    """
    if scaffold_column not in df.columns:
        raise KeyError(f"Column '{scaffold_column}' not found in DataFrame")

    s = (
        df[scaffold_column]
        .astype(str, copy=False)
        .str.strip()
        .replace({"None": "", "nan": ""})
    )

    # Пустые строки — молекулы без скаффолда
    empty_mask = s.eq("")
    s_non_empty = s[~empty_mask]

    total_compounds = int(len(df)) if count_empty_as_no_scaffold else int(len(s_non_empty))
    if total_compounds == 0:
        return CSRResult([0.0], [0.0], 0.0, None, 0, 0)

    scaffold_counts = Counter(s_non_empty)
    total_scaffolds = int(len(scaffold_counts))

    if total_scaffolds == 0:
        # Нет ни одного скаффолда в наборе
        return CSRResult([0.0, 1.0], [0.0, 0.0], 0.0, None, total_compounds, 0)

    # Сортировка по убыванию частоты
    sorted_scaffolds = sorted(scaffold_counts.items(), key=lambda x: x[1], reverse=True)

    cumulative_compounds: List[float] = []
    cumulative_scaffolds: List[float] = []
    compounds_covered = 0
    scaffolds_used = 0

    for _, count in sorted_scaffolds:
        scaffolds_used += 1
        compounds_covered += count
        cumulative_scaffolds.append(scaffolds_used / total_scaffolds)
        cumulative_compounds.append(compounds_covered / total_compounds)

    scaffold_fraction = [0.0] + cumulative_scaffolds
    compound_fraction = [0.0] + cumulative_compounds

    auc = float(np.trapz(compound_fraction, scaffold_fraction))

    # Находим F50 — минимальную долю скаффолдов для покрытия 50% соединений
    f50: Optional[float] = None
    for i, cf in enumerate(compound_fraction):
        if cf >= 0.5:
            f50 = float(scaffold_fraction[i])
            break

    return CSRResult(
        scaffold_fraction=scaffold_fraction,
        compound_fraction=compound_fraction,
        auc=auc,
        f50=f50,
        total_compounds=total_compounds,
        total_scaffolds=total_scaffolds,
    )


def plot_csr_curve(
    csr: CSRResult,
    label: str = "Carcinogenicity",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    base_color: Tuple[float, float, float] = (0/255, 128/255, 128/255),   # teal
    fill_alpha: float = 0.35,
    violet_color: Tuple[float, float, float] = (148/255, 0/255, 211/255), # фиолетовый
    tick_step: float = 0.1,
    title: str = "CSR (Cumulative Scaffold Recovery) curve",
) -> Tuple[Figure, Axes]:
    """
    Рисует CSR-кривую с заливкой и легендой (label, AUC, F50). Точка F50 — фиолетовая.

    Parameters
    ----------
    csr : CSRResult
        Результаты расчёта CSR.
    label : str, default 'Carcinogenicity'
        Заголовок набора в легенде.
    ax : matplotlib.axes.Axes, optional
        Существующая ось для отрисовки. Если None — создаётся новая фигура/ось.
    figsize : tuple, default (8, 6)
        Размер фигуры, если ax не передан.
    base_color : tuple, default teal
        Цвет линии кривой.
    fill_alpha : float, default 0.35
        Прозрачность заливки.
    violet_color : tuple, default dark violet
        Цвет маркера F50.
    tick_step : float, default 0.1
        Шаг основных делений по осям X/Y.
    title : str, default 'CSR (Cumulative Scaffold Recovery) curve'
        Заголовок графика.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    light_teal = tuple(np.array(base_color) * 0.8 + np.array([1, 1, 1]) * 0.2)

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True
    else:
        fig = ax.figure

    # Линия
    (line,) = ax.plot(csr.scaffold_fraction, csr.compound_fraction, color=base_color, linewidth=2)
    # Заливка
    ax.fill_between(csr.scaffold_fraction, csr.compound_fraction, color=light_teal, alpha=fill_alpha)

    # Точка F50
    f50_handle = None
    if csr.f50 is not None:
        (f50_handle,) = ax.plot([csr.f50], [0.5], "o", color=violet_color, markersize=8)

    # Оформление
    ax.set_xlabel("Fraction of scaffolds", fontsize=12)
    ax.set_ylabel("Fraction of compounds", fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Легенда
    legend_lines = [f"{label}", f"AUC = {csr.auc:.3f}"]
    if csr.f50 is not None:
        legend_lines.append(f"F50 = {csr.f50:.3f}")
    legend_text = "\n".join(legend_lines)

    text_handle = plt.Line2D([], [], color="none", label=legend_text)

    if f50_handle is not None:
        ax.legend(
            handles=[text_handle, line, f50_handle],
            labels=[legend_text, "CSR curve", "F50 point"],
            loc="lower right",
            fontsize=10,
        )
    else:
        ax.legend(
            handles=[text_handle, line],
            labels=[legend_text, "CSR curve"],
            loc="lower right",
            fontsize=10,
        )

    if created:
        fig.tight_layout()

    return fig, ax

