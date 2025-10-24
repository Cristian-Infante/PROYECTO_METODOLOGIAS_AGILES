# components/kpis.py
"""
Tarjetas KPI con Streamlit.

Tarea 2.3: Diseñar tarjetas KPI con st.metric()
- 3 columnas con st.columns(3)
- Métricas: "Total Trámites", "Promedio Mensual", "Variación Anual"
- Formato numérico con separadores de miles
- Emojis relevantes (📊, 📈, 📉)
- DoD: tarjetas se muestran con datos dinámicos y formato profesional
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st


@dataclass(frozen=True)
class KpiValues:
    total_tramites: float
    promedio_mensual: float
    variacion_anual_pct: float  # e.g., 12.3 == +12.3%


def _fmt_number(n: float) -> str:
    """Formatea números con separadores de miles, sin notación científica."""
    if n is None:
        return "—"
    # Si es entero, no mostrar decimales
    if float(n).is_integer():
        return f"{int(n):,}".replace(",", ".")  # Usa punto como separador de miles
    # Si tiene decimales, limitar a 2
    return f"{n:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")


def _fmt_percent(p: float) -> str:
    if p is None:
        return "—"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.2f}%"


def render_kpi_cards(values: KpiValues,
                     labels: Tuple[str, str, str] = ("Total Trámites", "Promedio Mensual", "Variación Anual"),
                     emojis: Tuple[str, str, str] = ("📊", "📈", "📉"),
                     help_texts: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)) -> None:
    """Renderiza 3 tarjetas KPI en columnas usando st.metric().
    
    Args:
        values: KpiValues con total, promedio mensual y variación anual (%).
        labels: Etiquetas para cada métrica.
        emojis: Emojis mostrados antes de cada etiqueta.
        help_texts: Tooltips opcionales (icono de ayuda) por métrica.
    """
    col1, col2, col3 = st.columns(3)

    # 1) Total de trámites
    with col1:
        label = f"{emojis[0]} {labels[0]}"
        st.metric(
            label=label,
            value=_fmt_number(values.total_tramites),
            help=help_texts[0]
        )

    # 2) Promedio mensual
    with col2:
        label = f"{emojis[1]} {labels[1]}"
        st.metric(
            label=label,
            value=_fmt_number(values.promedio_mensual),
            help=help_texts[1]
        )

    # 3) Variación anual (%)
    with col3:
        # delta en st.metric tiene flecha y color automático
        label = f"{emojis[2]} {labels[2]}"
        delta_value = values.variacion_anual_pct
        # Para coherencia visual, mostramos el valor base como "—" y la variación en delta
        st.metric(
            label=label,
            value=_fmt_percent(delta_value),
            delta=None,  # No necesitamos delta extra; ya va formateado en value
            help=help_texts[2]
        )
# components/kpis.py
import math

def _fmt_percent(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.2f}%"

