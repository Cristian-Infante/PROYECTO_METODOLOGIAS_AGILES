# Streamlit — carga Parquet preprocesado y visualiza.

import os
import math
import itertools
import pandas as pd
import altair as alt
import pydeck as pdk
import streamlit as st

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

# ---------- KPIs ----------
st.subheader("Indicadores")
c1, c2, c3 = st.columns(3)

total = len(df_f)
prom  = df_f["fecha_mes"].value_counts().mean() if df_f["fecha_mes"].notna().any() else float("nan")

# Variación entre el primer y último año seleccionados (si hay >=2)
anios_con_datos = sorted(df_f["anio"].dropna().unique().astype(int).tolist())
if len(anios_con_datos) >= 2:
    y0, y1 = anios_con_datos[0], anios_con_datos[-1]
    var = pct_change_between_two_years(df_f, y0, y1, meses_sel)
    label_var = f"Variación {y0}→{y1}"
else:
    var = None
    label_var = "Variación"

c1.metric("Total",         f"{total:,}".replace(",", "."))
c2.metric("Prom. mensual", f"{0 if math.isnan(prom) else round(prom):,}".replace(",", "."))
c3.metric(label_var, "N/D" if var is None else f"{var:.1f}%")

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
           .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
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
