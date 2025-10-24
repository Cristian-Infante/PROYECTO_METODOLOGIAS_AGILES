# Streamlit — carga Parquet preprocesado y visualiza.

import os
import math
import itertools
import pandas as pd
import altair as alt
import pydeck as pdk
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from components.kpis import render_kpi_cards, KpiValues

# ---------- Config ----------
st.set_page_config(page_title="Trámites Visibles", layout="wide")
PARQUET_PATH = "data/suit_tramites.parquet"

@st.cache_data(ttl=3600, show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

if not os.path.exists(PARQUET_PATH):
    st.error("No se encontró el Parquet. Ejecuta `python preprocess_suit.py`.")
    st.stop()

df = load_data(PARQUET_PATH)
if df.empty:
    st.warning("El Parquet está vacío.")
    st.stop()

st.title("Trámites Visibles")
st.caption("Serie mensual por año (superpuesta). Base: fecha de actualización.")

# ---------- util ----------
MESES_MAP = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
MESES_ORD = list(range(1,13))

def detect_peaks(series: pd.Series, threshold: float = 1.5) -> pd.Series:
    """
    Detecta picos/anomalías usando desviación estándar.

    Args:
        series: Serie de datos numéricos (trámites)
        threshold: Multiplicador de desv. estándar (1.5 = moderado, 2.0 = estricto)

    Returns:
        Serie booleana: True donde hay anomalía
    """
    if series.empty or series.std() == 0:
        return pd.Series(False, index=series.index)
    mean = series.mean()
    std = series.std()
    return (series - mean).abs() > (threshold * std)

def pct_change_between_two_years(d: pd.DataFrame, y0: int, y1: int, months: list[int]) -> float | None:
    """Variación % entre y0 (base) y y1 (comparado) usando SOLO meses presentes en ambos años."""
    if d.empty or y0 is None or y1 is None or y0 == y1:
        return None
    months = list(months) if months else MESES_ORD
    m0 = set(d.loc[d["anio"] == y0, "mes_num"].dropna().astype(int)) & set(months)
    m1 = set(d.loc[d["anio"] == y1, "mes_num"].dropna().astype(int)) & set(months)
    inter = sorted(m0 & m1)
    if not inter:
        return None
    a = len(d[(d["anio"] == y1) & (d["mes_num"].isin(inter))])
    b = len(d[(d["anio"] == y0) & (d["mes_num"].isin(inter))])
    if b == 0:
        return None
    return (a - b) / b * 100.0

# ---------- Sidebar (filtros dependientes) ----------
with st.sidebar:
    st.header("Filtros")

    # 1) Años
    anios_all = sorted(df["anio"].dropna().unique().astype(int).tolist())
    anios_sel = st.multiselect("Años", options=anios_all, default=anios_all, key="f_anios")
    df1 = df[df["anio"].isin(anios_sel)] if anios_sel else df.copy()

    # 2) Meses (derivados de años)
    meses_opts = sorted(df1["mes_num"].dropna().unique().astype(int).tolist()) or MESES_ORD
    meses_sel = st.multiselect("Meses", options=meses_opts, default=meses_opts,
                               format_func=lambda m: MESES_MAP.get(m, str(m)), key="f_meses")
    df2 = df1[df1["mes_num"].isin(meses_sel)] if meses_sel else df1.copy()

    # 3) Departamento
    deptos_opts = sorted(df2["departamento"].dropna().unique().tolist())
    d_sel = st.multiselect("Departamento", options=deptos_opts, default=[], key="f_depto")
    df3 = df2[df2["departamento"].isin(d_sel)] if d_sel else df2.copy()

    # 4) Municipio
    municipios_opts = sorted(df3["municipio"].dropna().unique().tolist())
    mpio_sel = st.multiselect("Municipio", options=municipios_opts, default=[], key="f_mpio")
    df4 = df3[df3["municipio"].isin(mpio_sel)] if mpio_sel else df3.copy()

    # 5) Entidad
    entidades_opts = sorted(df4["entidad"].dropna().unique().tolist())
    entidades_opts = entidades_opts[:300] if len(entidades_opts) > 300 else entidades_opts
    e_sel = st.multiselect("Entidad", options=entidades_opts, default=[], key="f_entidad")
    df_f = df4[df4["entidad"].isin(e_sel)] if e_sel else df4.copy()

    # --- Controles de análisis visual ---
    st.divider()
    st.subheader("Análisis Visual")
    mostrar_tendencia = st.checkbox("Línea de tendencia", value=False, key="toggle_tendencia")
    mostrar_anomalias = st.checkbox("Detectar anomalías", value=False, key="toggle_anomalias")

    threshold = 1.5
    if mostrar_anomalias:
        threshold = st.slider(
            "Sensibilidad de detección",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Valores más bajos detectan más anomalías. 1.5=moderado, 2.0=estricto",
            key="threshold_slider"
        )

# ---------- KPIs ----------
st.subheader("Indicadores")
# Total y promedio mensual (idéntico a tu lógica actual)
total = len(df_f)
prom  = df_f["fecha_mes"].value_counts().mean() if df_f["fecha_mes"].notna().any() else float("nan")

# --- Variación robusta: escoger par de años con meses en común y base > 0 ---
anios_con_datos = sorted(df_f["anio"].dropna().unique().astype(int).tolist())

def _yoy_variation_robusta(d: pd.DataFrame, meses_sel: list[int] | None):
    if d.empty or d["anio"].isna().all() or d["mes_num"].isna().all():
        return None, None, None  # (var, y0, y1)

    meses = list(meses_sel) if meses_sel else MESES_ORD

    # Conteos por año/mes (solo meses válidos)
    base = (
        d[d["mes_num"].isin(meses)]
        .groupby(["anio", "mes_num"], as_index=False)
        .size()
        .rename(columns={"size": "tramites"})
    )
    if base.empty:
        return None, None, None

    # Mapa de meses disponibles por año
    meses_por_anio = (
        base.groupby("anio")["mes_num"]
            .apply(lambda s: set(s.astype(int).tolist()))
            .to_dict()
    )

    # Intentar pares (y0, y1) con intersección de meses y base > 0
    anios = sorted(meses_por_anio.keys())
    mejor = None  # (var, y0, y1)
    for y1 in reversed(anios):          # priorizar el más reciente como comparado
        for y0 in anios:                # base
            if y0 >= y1:
                continue
            inter = meses_por_anio[y0] & meses_por_anio[y1]
            if not inter:
                continue

            a = int(base[(base["anio"] == y1) & (base["mes_num"].isin(inter))]["tramites"].sum())
            b = int(base[(base["anio"] == y0) & (base["mes_num"].isin(inter))]["tramites"].sum())
            if b == 0:
                continue

            var = (a - b) / b * 100.0
            mejor = (var, y0, y1)
            break
        if mejor:
            break

    # Si no encontramos ningún par válido, intentar y1 vs y1-1 (año inmediato anterior)
    if not mejor and len(anios) >= 2:
        y1 = anios[-1]
        y0 = y1 - 1
        if y0 in meses_por_anio:
            inter = meses_por_anio[y0] & meses_por_anio[y1]
            if inter:
                a = int(base[(base["anio"] == y1) & (base["mes_num"].isin(inter))]["tramites"].sum())
                b = int(base[(base["anio"] == y0) & (base["mes_num"].isin(inter))]["tramites"].sum())
                if b > 0:
                    var = (a - b) / b * 100.0
                    mejor = (var, y0, y1)

    return mejor if mejor else (None, None, None)

var, y0, y1 = _yoy_variation_robusta(df_f, meses_sel)
label_var = f"Variación {y0}→{y1}" if (y0 is not None and y1 is not None) else "Variación Anual"


# Empaquetar para el componente
values = KpiValues(
    total_tramites=total,
    promedio_mensual=0 if (isinstance(prom, float) and math.isnan(prom)) else float(prom),
    variacion_anual_pct=var if (var is not None and not (isinstance(var, float) and math.isnan(var))) else None,
)


# Renderizar 3 tarjetas con emojis y formato profesional
render_kpi_cards(
    values,
    labels=("Total Trámites", "Promedio Mensual", label_var),
    help_texts=(
        "Cantidad acumulada de trámites en el periodo filtrado.",
        "Promedio de trámites por mes.",
        "Variación respecto al periodo equivalente del año anterior.",
    ),
)
st.divider()

# ---------- Serie mensual superpuesta por año ----------
st.subheader("Evolución mensual por año (superpuesta)")

if df_f["anio"].notna().any() and df_f["mes_num"].notna().any():
    base = (
        df_f.groupby(["anio", "mes_num"], as_index=False)
            .size()
            .rename(columns={"size": "tramites"})
    )
    years_for_grid = sorted(base["anio"].dropna().unique().astype(int).tolist())
    months_for_grid = sorted(set(base["mes_num"].dropna().astype(int)) & set(meses_sel)) if meses_sel else MESES_ORD
    grid = pd.DataFrame(list(itertools.product(years_for_grid, months_for_grid)), columns=["anio", "mes_num"])
    serie = (
        grid.merge(base, on=["anio", "mes_num"], how="left")
            .assign(tramites=lambda d: d["tramites"].fillna(0).astype(int))
            .sort_values(["anio", "mes_num"])
    )
    # columna de texto de mes para tooltip (evita lambda en format)
    serie["mes_nombre"] = serie["mes_num"].map(MESES_MAP)

    # Detectar anomalías por año si está activado
    if mostrar_anomalias:
        serie["es_anomalia"] = False
        for anio in years_for_grid:
            mask_anio = serie["anio"] == anio
            serie.loc[mask_anio, "es_anomalia"] = detect_peaks(
                serie.loc[mask_anio, "tramites"], 
                threshold=threshold
            )

    # Gráfico base
    chart = (
        alt.Chart(serie)
           .mark_line(point=True)
           .encode(
               x=alt.X("mes_num:O", title="Mes",
                       sort=MESES_ORD,
                       axis=alt.Axis(labelExpr="['','Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'][datum.value]")),
               y=alt.Y("tramites:Q", title="Trámites"),
               color=alt.Color("anio:O", title="Año"),
               tooltip=[alt.Tooltip("anio:O", title="Año"),
                        alt.Tooltip("mes_nombre:N", title="Mes"),
                        alt.Tooltip("tramites:Q", title="Trámites")]
           )
           .properties(height=360)
    )

    # Capa de anomalías
    if mostrar_anomalias:
        anomalias = serie[serie["es_anomalia"]]
        if not anomalias.empty:
            chart_anomalias = (
                alt.Chart(anomalias)
                   .mark_point(size=200, shape="diamond", filled=True)
                   .encode(
                       x=alt.X("mes_num:O"),
                       y=alt.Y("tramites:Q"),
                       color=alt.value("#FF4B4B"),
                       tooltip=[
                           alt.Tooltip("anio:O", title="Año"),
                           alt.Tooltip("mes_nombre:N", title="Mes"),
                           alt.Tooltip("tramites:Q", title="Trámites"),
                           alt.Tooltip("es_anomalia:N", title="Anomalía")
                       ]
                   )
            )
            chart = chart + chart_anomalias

    # Línea de tendencia por año (limitada al rango de datos)
    if mostrar_tendencia:
        tendencias = []
        for anio in years_for_grid:
            datos_anio = serie[serie["anio"] == anio].copy()
            if len(datos_anio) >= 2:
                # Calcular la regresión manualmente para controlar el rango


                X = datos_anio["mes_num"].values.reshape(-1, 1)
                y = datos_anio["tramites"].values

                model = LinearRegression()
                model.fit(X, y)

                # Predecir solo para los meses que tienen datos
                datos_anio["tendencia"] = model.predict(X)

                chart_tendencia = (
                    alt.Chart(datos_anio)
                       .mark_line(strokeDash=[5, 5], size=2, opacity=0.6)
                       .encode(
                           x=alt.X("mes_num:O"),
                           y=alt.Y("tendencia:Q"),
                           color=alt.Color("anio:O", legend=None)
                       )
                )
                tendencias.append(chart_tendencia)

        for tend in tendencias:
            chart = chart + tend

    st.altair_chart(chart.interactive(), use_container_width=True)

    # Mostrar resumen de anomalías si están activadas
    if mostrar_anomalias and "es_anomalia" in serie.columns:
        num_anomalias = serie["es_anomalia"].sum()
        if num_anomalias > 0:
            st.info(f"Se detectaron **{num_anomalias}** anomalías con threshold={threshold}")
        else:
            st.success(f"✅ No se detectaron anomalías con threshold={threshold}")
else:
    st.info("No hay datos temporales para graficar.")

st.divider()

# ---------- Mapa ----------
st.subheader("Distribución geográfica")

coords_mask = df_f["coords_validas"] if "coords_validas" in df_f.columns else pd.Series(False, index=df_f.index)
geo = (
    df_f[coords_mask]
      .dropna(subset=["lat", "lon"])
      .groupby(["departamento", "municipio", "lat", "lon"], as_index=False)
      .size()
      .rename(columns={"size": "tramites"})
)

if geo.empty:
    st.info("No hay coordenadas válidas para este filtro.")
else:
    geo["radius"] = (geo["tramites"] * 30).clip(lower=2000).astype(float)
    geo["lat"] = geo["lat"].astype(float)
    geo["lon"] = geo["lon"].astype(float)

    view = pdk.ViewState(latitude=4.570868, longitude=-74.297333, zoom=4.8, pitch=0, bearing=0)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=geo,
        get_position='[lon, lat]',
        get_radius='radius',
        get_fill_color='[0, 180, 255, 200]',
        get_line_color='[255, 255, 255, 220]',
        stroked=True,
        line_width_min_pixels=1,
        radius_min_pixels=2,
        radius_max_pixels=120,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {"text": "Depto: {departamento}\nMunicipio: {municipio}\nTrámites: {tramites}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))

# ---------- Tabla resumen ----------
with st.expander("Tabla por año/mes"):
    if df_f["anio"].notna().any() and df_f["mes_num"].notna().any():
        tabla = (
            df_f.groupby(["anio", "mes_num"], as_index=False)
                .size()
                .rename(columns={"size": "tramites", "mes_num": "mes"})
                .sort_values(["anio", "mes"])
        )
        st.dataframe(tabla, use_container_width=True, hide_index=True)
    else:
        st.write("Sin datos temporales para mostrar.")
