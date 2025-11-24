import base64
import sys, json, time, sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Config global ----------
st.set_page_config(page_title="🦷 Consultorio Dental", layout="wide")
sns.set_style("white")
sns.set_palette("muted")

# Ocultar menú y footer de Streamlit

# ---------- Paths & auth ----------
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from auth import check_credentials                                       # noqa: E402

MODELS_DIR  = project_root / "models" / "models"
DATA_DIR    = project_root / "data"
REPORTS_DIR = MODELS_DIR / "reportes"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helper ----------
def toast(msg:str, icon:str=""):
    """Intenta usar st.toast (v≥1.25); fallback a st.success."""
    try: st.toast(msg, icon=icon)
    except AttributeError: st.success(msg)

# ---------- Session defaults ----------
defaults = {
    "authenticated": False, "username": "",
    "data_loaded": False,   "df": None,
    "crud": {}, "show_crud": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ---------- Login ----------
if not st.session_state.authenticated:

    # ---- CSS para compactar el formulario y centrarlo ----
    st.markdown(
        """
        <style>
        /* Form container */
        [data-testid="stForm"] {
            max-width: 320px;                      /* ancho fijo */
            margin: 0 auto;                        /* centra horizontal */
            padding: 1.5rem 2rem 2rem;
            border: 1px solid #3a3a3a;             /* borde tenue (tema dark) */
            border-radius: 0.75rem;
            background: #1e1e1e;                   /* mismo tono que sidebar */
            box-shadow: 0 4px 12px rgba(0,0,0,0.45);
        }
        /* Inputs más pequeños */
        [data-testid="stForm"] input {
            padding: 0.4rem 0.6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---- Columna central para logo + formulario ----
    _, col_center, _ = st.columns([2, 3, 2])
    with col_center:

        # Logo circular desde assets/logo.png
        logo_path = Path("assets/logo.png")
        if logo_path.exists():
            encoded = base64.b64encode(logo_path.read_bytes()).decode()
            st.markdown(
                f"""
                <div style="display:flex;justify-content:center;margin-bottom:1rem;">
                    <img src="data:image/png;base64,{encoded}"
                         style="width:150px;height:150px;border-radius:50%;object-fit:cover;"/>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---- Formulario de acceso ----
        with st.form("login_form"):
            st.markdown("<h3 style='text-align:center;margin-top:0;'>Iniciar sesión</h3>",
                        unsafe_allow_html=True)

            usuario = st.text_input("Usuario")
            contr  = st.text_input("Contraseña", type="password")
            entrar = st.form_submit_button("Entrar")

            if entrar:
                if check_credentials(usuario, contr):
                    st.session_state.authenticated = True
                    st.session_state.username = usuario
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")

    st.stop()

# ---------- Utils ----------
@st.cache_data(show_spinner=False)
def load_data(areas_io, pac_io, trat_io, aten_io):
    """Carga, fusiona y enriquece los 4 CSV tal como en el script original."""
    areas  = pd.read_csv(areas_io, header=None, names=['area_id','area_name'])
    pac    = pd.read_csv(pac_io, header=None,
                         names=['patient_id','patient_name','age','gender','registration_date'])
    trat   = pd.read_csv(trat_io, header=None,
                         names=['treatment_id','treatment_name','area_id','price'])
    aten   = pd.read_csv(aten_io, header=None,
                         names=['attention_id','patient_id','treatment_id',
                                'attention_date','payment_type','status','extra']
                        ).drop(columns=['extra'], errors='ignore')
    df = (aten.merge(pac,  on='patient_id', how='left')
               .merge(trat, on='treatment_id', how='left')
               .merge(areas,on='area_id',     how='left'))
    df['attention_date'] = pd.to_datetime(df['attention_date'])
    df['year_month']     = df['attention_date'].dt.to_period('M').dt.to_timestamp()
    df['year']           = df['attention_date'].dt.year
    return df

def reset_data():
    st.session_state.data_loaded = False
    st.session_state.df = None

# ---------- Top bar ----------
c1, c2 = st.columns([10,1])
c1.markdown(f"## Bienvenido, **{st.session_state.username}**")
if c2.button("🔄", help="Volver a ingresar datos"):
    reset_data()
    st.rerun()

# ---------- Pestañas principales ----------
tabs = st.tabs(["🏠 Inicio", "📥 Carga", "📊 Dashboard", "📈 Predicción"])
# ========================= FIN PARTE 1 / 4 =========================
# ========================= sistema_web_full.py — PARTE 2 / 4 =========================
# (Añade esta sección inmediatamente después de la definición de `tabs`)
# ---------- Pestaña 0 : Inicio ----------
# ---------- Pestaña 0 : Inicio ----------
# ---------- Pestaña 0 : Inicio ----------
# ---------- Pestaña 0 : Inicio ----------
# ---------- Pestaña 0 : Inicio ----------
with tabs[0]:
    st.subheader("Información general del consultorio")

    # Datos fijos (áreas)
    areas_df = pd.DataFrame([
        {"ID": 1, "Área": "Odontología General"},
        {"ID": 2, "Área": "Ortodoncia"},
        {"ID": 3, "Área": "Endodoncia"},
        {"ID": 4, "Área": "Periodoncia"},
        {"ID": 5, "Área": "Odontopediatría"},
        {"ID": 6, "Área": "Prostodoncia"},
        {"ID": 7, "Área": "Cirugía Oral"},
    ])

    # Datos fijos (tratamientos + costo)
    tratamientos_df = pd.DataFrame([
        {"Tratamiento": "Consulta General",                                   "Área": "Odontología General", "Costo": 60},
        {"Tratamiento": "Limpieza Dental",                                   "Área": "Odontología General", "Costo": 180},
        {"Tratamiento": "Tratamiento de Caries",                             "Área": "Odontología General", "Costo": 200},
        {"Tratamiento": "Colocación de Brackets",                            "Área": "Ortodoncia",          "Costo": 3000},
        {"Tratamiento": "Mantenimiento de Ortodoncia",                       "Área": "Ortodoncia",          "Costo": 200},
        {"Tratamiento": "Tratamiento de Conducto",                           "Área": "Endodoncia",          "Costo": 150},
        {"Tratamiento": "Retratamiento de Conducto",                         "Área": "Endodoncia",          "Costo": 300},
        {"Tratamiento": "Tratamiento de Encías",                             "Área": "Periodoncia",         "Costo": 200},
        {"Tratamiento": "Cirugía de Encías",                                 "Área": "Periodoncia",         "Costo": 200},
        {"Tratamiento": "Consulta Infantil",                                 "Área": "Odontopediatría",     "Costo": 60},
        {"Tratamiento": "Aplicación de Flúor",                               "Área": "Odontopediatría",     "Costo": 20},
        {"Tratamiento": "Prótesis Dental Parcial",                           "Área": "Prostodoncia",        "Costo": 800},
        {"Tratamiento": "Prótesis Completa",                                 "Área": "Prostodoncia",        "Costo": 600},
        {"Tratamiento": "Corona de Oro",                                     "Área": "Prostodoncia",        "Costo": 2000},
        {"Tratamiento": "Corona de Cromo",                                   "Área": "Prostodoncia",        "Costo": 250},
        {"Tratamiento": "Corona Jacket",                                     "Área": "Prostodoncia",        "Costo": 200},
        {"Tratamiento": "Perno Muñón",                                       "Área": "Prostodoncia",        "Costo": 120},
        {"Tratamiento": "Prótesis Fija Canino a Canino (pieza de cromo)",    "Área": "Prostodoncia",        "Costo": 1400},
        {"Tratamiento": "Extracción de Muelas del Juicio",                   "Área": "Cirugía Oral",        "Costo": 500},
    ])

    # Métricas rápidas
    c1, c2, c3 = st.columns(3)
    c1.metric("Áreas totales",        len(areas_df))
    c2.metric("Tratamientos totales", len(tratamientos_df))
    c3.metric("Rango de costos",
              f"{tratamientos_df['Costo'].min():,.0f} – {tratamientos_df['Costo'].max():,.0f}")

    # Tablas
    st.markdown("### Áreas")
    st.table(areas_df)

    st.markdown("### Tratamientos y costos")
    st.table(tratamientos_df)


# ---------- Pestaña 0 : Carga de datos ----------
with tabs[1]:
    st.subheader("Carga de archivos CSV")

    # Controles de subida centrados
    cA, cB, cC = st.columns([1,2,1])
    with cB:
        areas_f        = st.file_uploader("Áreas (CSV)",        type="csv", on_change=reset_data, key="up1")
        pacientes_f    = st.file_uploader("Pacientes (CSV)",    type="csv", on_change=reset_data, key="up2")
        tratamientos_f = st.file_uploader("Tratamientos (CSV)", type="csv", on_change=reset_data, key="up3")
        atenciones_f   = st.file_uploader("Atenciones (CSV)",   type="csv", on_change=reset_data, key="up4")

    ready = all([areas_f, pacientes_f, tratamientos_f, atenciones_f])

    # ---------- Progreso animado (igual que el original) ----------
    if ready and not st.session_state.data_loaded:
        msg = st.empty()
        progress = st.progress(0)
        for i in range(10):
            if   i < 3: msg.info("Procesando archivos…")
            elif i < 6: msg.info("Limpiando datos…")
            elif i < 9: msg.info("Optimizando tablas…")
            else:       msg.info("¡Ya casi está listo!")
            time.sleep(1)                       # conserva sensación “paso a paso”
            progress.progress((i+1)/10)
        msg.empty()

        # Carga real
        st.session_state.df = load_data(areas_f, pacientes_f, tratamientos_f, atenciones_f)
        st.session_state.data_loaded = True
        progress.empty()
        toast("Datos cargados ✔️", "✅")

    # ---------- Vista previa ----------
    if st.session_state.data_loaded:
        st.markdown("### Vista rápida (primeros 20 registros)")
        st.dataframe(st.session_state.df.head(20), use_container_width=True)
# ========================= FIN PARTE 2 / 4 =========================
# ========================= sistema_web_full.py — PARTE 3 / 4 =========================
# ---------- Funciones de filtro y visualización (Dashboard) ----------

def apply_filters(data: pd.DataFrame):
    """Devuelve DataFrame filtrado + nivel de agregación elegido (idéntico al original)."""
    default_min, default_max = data['attention_date'].min(), data['attention_date'].max()

    start_date, end_date = st.sidebar.date_input(
        "Rango de fechas", [default_min, default_max],
        min_value=default_min, max_value=default_max, key='date_filter')

    areas   = st.sidebar.multiselect("Áreas",   data['area_name'].unique().tolist(),
                                     default=list(data['area_name'].unique()),   key='area_filter')
    genders = st.sidebar.multiselect("Géneros", data['gender'].unique().tolist(),
                                     default=list(data['gender'].unique()),      key='gender_filter')
    payments= st.sidebar.multiselect("Tipo de pago", data['payment_type'].unique().tolist(),
                                     default=list(data['payment_type'].unique()), key='payment_filter')
    status  = st.sidebar.multiselect("Estado",  data['status'].unique().tolist(),
                                     default=list(data['status'].unique()),      key='status_filter')
    agg_lvl = st.sidebar.selectbox("Agrupar evolución por", ['Diario','Mensual','Anual'],
                                   index=1, key='agg_filter')

    df_f = data[
        (data['attention_date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))) &
        (data['area_name'].isin(areas)) &
        (data['gender'].isin(genders)) &
        (data['payment_type'].isin(payments)) &
        (data['status'].isin(status))
    ]
    return df_f, agg_lvl


def plot_area_attention(data):
    fig, ax = plt.subplots(figsize=(8,5))
    order = data['area_name'].value_counts().index
    sns.countplot(data=data, y='area_name', order=order, ax=ax)
    ax.set_title("Atenciones por Área"); ax.set_xlabel("Cantidad"); ax.set_ylabel("")
    return fig


def plot_top_treatments(data):
    fig, ax = plt.subplots(figsize=(8,5))
    top10 = data['treatment_name'].value_counts().nlargest(10)
    sns.barplot(x=top10.values, y=top10.index, ax=ax)
    ax.set_title("Top 10 Tratamientos"); ax.set_xlabel("Cantidad"); ax.set_ylabel("")
    return fig


def plot_age_distribution(data):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(data['age'], bins=20, kde=True, ax=ax)
    ax.set_title("Distribución de Edad"); ax.set_xlabel("Edad"); ax.set_ylabel("Frecuencia")
    return fig


def plot_time_series(data, level):
    if   level=='Diario':
        series = data.groupby('attention_date').size()
        xlabel = 'Fecha'
    elif level=='Mensual':
        series = data.groupby('year_month').size()
        xlabel = 'Mes'
    else:
        series = data.groupby('year').size()
        xlabel = 'Año'
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(series.index, series.values)
    ax.set_title(f"Evolución ({level})"); ax.set_xlabel(xlabel); ax.set_ylabel("Cantidad")
    return fig


def plot_payment_status(data):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    data['payment_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
    ax1.set_title("Tipo de Pago"); ax1.set_ylabel("")
    data['status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2)
    ax2.set_title("Estado de Pago"); ax2.set_ylabel("")
    return fig


def plot_heatmap(data):
    pivot = data.pivot_table(
        index='area_name', columns='treatment_name',
        values='attention_id', aggfunc='count', fill_value=0
    )
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(pivot, cmap='YlGnBu', linewidths=.5, ax=ax)
    ax.set_title("Heatmap: Área vs Tratamiento"); ax.set_xlabel(""); ax.set_ylabel("")
    return fig


# ---------- Pestaña 1 : Dashboard Descriptivo ----------
with tabs[2]:
    if not st.session_state.data_loaded:
        st.info("Sube los archivos en la pestaña *Carga* para visualizar el dashboard.")
        st.stop()

    df = st.session_state.df
    st.sidebar.title(f"Usuario: {st.session_state.username}")
    st.title("Dashboard de Atenciones Dentales")

    st.sidebar.header("Filtros descriptivos")
    df_f, agg_level = apply_filters(df)

    # ---- Render de gráficas ----
    st.subheader("Atenciones por Área")
    st.pyplot(plot_area_attention(df_f))

    st.subheader("Top 10 Tratamientos")
    st.pyplot(plot_top_treatments(df_f))

    st.subheader("Distribución de Edad")
    st.pyplot(plot_age_distribution(df_f))

    st.subheader(f"Evolución {agg_level}")
    st.pyplot(plot_time_series(df_f, agg_level))

    st.subheader("Tipo de Pago y Estado")
    st.pyplot(plot_payment_status(df_f))

    st.subheader("Heatmap de Atención")
    st.pyplot(plot_heatmap(df_f))

with tabs[3]:
    if not st.session_state.data_loaded:
        st.info("Carga primero los datos para generar predicciones.")
        st.stop()

    df = st.session_state.df
    st.subheader("Predicción de ingresos mensuales")

    # ---------- FEATURE ENGINEERING (idéntico a original) ----------
    agg_pred = (
        df.groupby(df["attention_date"].dt.to_period("M"))
          .agg(ingreso_total      = ('price','sum'),
               total_atenciones   = ('attention_id','count'),
               pacientes_unicos   = ('patient_id', pd.Series.nunique),
               pct_pagado         = ('status', lambda s: np.mean(s=='Pagado')))
          .reset_index()
    )
    agg_pred['fecha_mes'] = pd.to_datetime(agg_pred['attention_date'].dt.to_timestamp())
    agg_pred.sort_values('fecha_mes', inplace=True, ignore_index=True)

    # Señales estacionales y retardos
    agg_pred['sin_mes']  = np.sin(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
    agg_pred['cos_mes']  = np.cos(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
    agg_pred['anio_cent'] = agg_pred['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()
    agg_pred['lag1'] = agg_pred['ingreso_total'].shift(1).bfill()
    agg_pred['lag2'] = agg_pred['ingreso_total'].shift(2).bfill()
    agg_pred['ma3']  = agg_pred['ingreso_total'].rolling(3).mean().shift(1).bfill()

    VARIABLES = ['total_atenciones','pacientes_unicos','pct_pagado',
                 'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3']

    model = joblib.load(MODELS_DIR / "modelo_final_2025_brutos.joblib")

    # ---------- Evaluación en subconjunto de test ----------
    _, test_idx = train_test_split(agg_pred.index, test_size=0.2, random_state=42)
    y_test   = agg_pred.loc[test_idx, 'ingreso_total']
    y_pred   = model.predict(agg_pred.loc[test_idx, VARIABLES])

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE",  f"{mean_absolute_error(y_test, y_pred):,.0f}")
    col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
    col3.metric("R²",   f"{r2_score(y_test, y_pred):.3f}")

    # ---------- Predicción puntual desde sidebar ----------
    st.sidebar.header("Predicción puntual")
    mes_sel = st.sidebar.selectbox("Mes (1 – 12)", list(range(1,13)), 0)
    if st.sidebar.button("Calcular predicción"):
        fila = agg_pred[agg_pred['fecha_mes'].dt.month == mes_sel].iloc[0]
        pred_val = float(model.predict(pd.DataFrame({v:[fila[v]] for v in VARIABLES})))
        toast(f"Ingreso estimado para mes {mes_sel}: {pred_val:,.0f}", "💰")

    # ---------- Histórico de ingresos ----------
    st.subheader("Histórico de ingresos (2021‑2024)")
    st.line_chart(agg_pred.set_index('fecha_mes')['ingreso_total'])

    # ---------- Pronóstico para 2025 ----------
    future = pd.DataFrame({'fecha_mes': pd.date_range('2025-01-01','2025-12-01',freq='MS')})
    future['sin_mes']  = np.sin(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
    future['cos_mes']  = np.cos(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
    future['anio_cent']= future['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()

    last_row = agg_pred.iloc[-1]
    for lag in ['lag1','lag2','ma3']:
        future[lag] = last_row[lag]

    grp = agg_pred.groupby(agg_pred['fecha_mes'].dt.month)
    future['total_atenciones'] = future['fecha_mes'].dt.month.map(grp['total_atenciones'].mean())
    future['pacientes_unicos'] = future['fecha_mes'].dt.month.map(grp['pacientes_unicos'].mean())
    future['pct_pagado']       = future['fecha_mes'].dt.month.map(grp['pct_pagado'].mean())

    future['predicted_income'] = model.predict(future[VARIABLES])

    st.subheader("Pronóstico 2025")
    st.dataframe(future[['fecha_mes','predicted_income']].reset_index(drop=True))
    st.line_chart(future.set_index('fecha_mes')['predicted_income'])

    # ---------- Exportación de reporte ----------
    if st.button("📤 Exportar reporte completo"):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path   = REPORTS_DIR / f"reporte_completo_{stamp}.csv"
        excel_path = csv_path.with_suffix('.xlsx')

        # CSV plano solo pronóstico
        future[['fecha_mes','predicted_income']].to_csv(csv_path, index=False)

        # Excel con todas las hojas
        with pd.ExcelWriter(excel_path) as writer:
            st.session_state.df.to_excel(writer, sheet_name="Atenciones_Completas", index=False)
            pd.DataFrame({
                'fecha_mes': agg_pred.loc[test_idx,'fecha_mes'],
                'actual_income': y_test.values,
                'predicted_income': y_pred
            }).to_excel(writer, sheet_name="Prediccion_vs_Real", index=False)
            future[['fecha_mes','predicted_income']].to_excel(writer, sheet_name="Pronostico_2025", index=False)

        toast(f"Archivos guardados: {csv_path.name} y Excel", "📄")

        # Guardar info para formulario CRUD
        st.session_state['crud'] = {
            'usuario': st.session_state.username,
            'archivo_csv': str(csv_path),
            'archivo_excel': str(excel_path),
            'parametros': json.dumps({'generado_por': st.session_state.username,
                                      'fecha': datetime.now().isoformat()}),
            'fecha_creacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.session_state['show_crud'] = True

    # ---------- Formulario CRUD ----------
    if st.session_state.get('show_crud', False):
        st.markdown("### Registrar reporte en la base de datos")
        with st.expander("Completa los datos del reporte", expanded=True):
            cd = st.session_state['crud']
            # Vista previa CSV
            if Path(cd['archivo_csv']).exists():
                st.markdown("**Vista previa del pronóstico (CSV):**")
                st.dataframe(pd.read_csv(cd['archivo_csv']).head(20))

            with st.form("form_prediccion"):
                nombre = st.text_input("Nombre del reporte", value=f"Reporte {cd['fecha_creacion'][:10]}")
                tipo   = st.selectbox("Tipo", ["Gráfica","Predicción","Ambos"], index=2)
                descripcion = st.text_area("Descripción")
                if st.form_submit_button("Guardar en BD"):
                    # Insertar en SQLite
                    db_path = DATA_DIR / "consultorio.db"
                    conn = sqlite3.connect(db_path)
                    cur  = conn.cursor()
                    cur.execute("""
                        INSERT INTO predicciones
                        (usuario, fecha_creacion, nombre, tipo, archivo_csv, parametros, descripcion)
                        VALUES (?,?,?,?,?,?,?)
                    """, (cd['usuario'], cd['fecha_creacion'], nombre, tipo,
                          cd['archivo_csv'], cd['parametros'], descripcion))
                    conn.commit(); conn.close()
                    toast("Registro guardado en base de datos ✔️", "💾")
                    st.session_state['show_crud'] = False

