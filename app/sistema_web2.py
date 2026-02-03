# ===================== PARTE 1/3 =====================
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import sqlite3

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Asegurar import de src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from auth import check_credentials

st.set_page_config(page_title="Consultorio Dental - Sistema Web", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

login_placeholder = st.empty()

if not st.session_state.authenticated:
    with login_placeholder.form("login_form"):
        st.title("🔐 Iniciar Sesión")
        user = st.text_input("Usuario", key="login_user")
        pwd  = st.text_input("Contraseña", type="password", key="login_pwd")
        submit = st.form_submit_button("Entrar")
        if submit:
            if check_credentials(user, pwd):
                st.session_state.authenticated = True
                st.session_state.username = user
            else:
                st.error("Usuario o contraseña incorrectos")
    if st.session_state.authenticated:
        login_placeholder.empty()
    else:
        st.stop()

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

def reset_data_loaded():
    st.session_state.data_loaded = False
    st.session_state.df = None

uploader_placeholder = st.empty()
dashboard_placeholder = st.empty()

st.markdown("---")
col_top = st.columns([2,1])
with col_top[0]:
    if st.button("🔄 Volver a ingresar los datos"):
        reset_data_loaded()
        st.experimental_rerun()
with col_top[1]:
    st.markdown(f"<div style='text-align:right;'><b>Usuario:</b> {st.session_state.get('username','')}</div>", unsafe_allow_html=True)
st.markdown("---")

with uploader_placeholder.container():
    st.header("Carga de datos CSV")
    cols = st.columns([1, 2, 1])
    with cols[1]:
        areas_file        = st.file_uploader("Áreas (CSV)", type="csv", key="up1", on_change=reset_data_loaded)
        pacientes_file    = st.file_uploader("Pacientes (CSV)", type="csv", key="up2", on_change=reset_data_loaded)
        tratamientos_file = st.file_uploader("Tratamientos (CSV)", type="csv", key="up3", on_change=reset_data_loaded)
        atenciones_file   = st.file_uploader("Atenciones (CSV)", type="csv", key="up4", on_change=reset_data_loaded)

if all([areas_file, pacientes_file, tratamientos_file, atenciones_file]) and not st.session_state.data_loaded:
    # Elemento dinámico para mostrar mensajes
    msg_placeholder = st.empty()
    progress = st.progress(0)
    for i in range(10):
        if i < 3:
            msg_placeholder.info("Procesando archivos...")
        elif i < 6:
            msg_placeholder.info("Limpiando archivos...")
        elif i < 9:
            msg_placeholder.info("Optimizando archivos...")
        else:
            msg_placeholder.info("¡Ya casi está listo!")
        time.sleep(1)
        progress.progress((i + 1) / 10)
    msg_placeholder.empty()
    def load_data(a_io, p_io, t_io, at_io):
        areas = pd.read_csv(a_io, header=None, names=['area_id','area_name'])
        pacientes = pd.read_csv(p_io, header=None,
                                names=['patient_id','patient_name','age','gender','registration_date'])
        tratamientos = pd.read_csv(t_io, header=None,
                                   names=['treatment_id','treatment_name','area_id','price'])
        atenciones = pd.read_csv(at_io, header=None,
                                  names=['attention_id','patient_id','treatment_id',
                                         'attention_date','payment_type','status','extra'])
        atenciones = atenciones.drop(columns=['extra'], errors='ignore')
        df = (atenciones
              .merge(pacientes, on='patient_id', how='left')
              .merge(tratamientos, on='treatment_id', how='left')
              .merge(areas, on='area_id', how='left'))
        df['attention_date'] = pd.to_datetime(df['attention_date'])
        df['year_month'] = df['attention_date'].dt.to_period('M').dt.to_timestamp()
        df['year'] = df['attention_date'].dt.year
        return df
    st.session_state.df = load_data(areas_file, pacientes_file, tratamientos_file, atenciones_file)
    st.session_state.data_loaded = True
    uploader_placeholder.empty()

if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    with dashboard_placeholder.container():
        st.sidebar.title(f"Usuario: {st.session_state.username}")

        st.title("Dashboard de Atenciones Dentales Personalizado")
        sns.set_style("whitegrid")

        st.sidebar.header("Filtros Descriptivos")
        def apply_filters(data):
            default_min = data['attention_date'].min()
            default_max = data['attention_date'].max()
            start_date, end_date = st.sidebar.date_input(
                "Rango de fechas", [default_min, default_max],
                min_value=default_min, max_value=default_max, key='date_filter')
            selected_areas = st.sidebar.multiselect("Áreas", data['area_name'].unique().tolist(),
                                                    default=list(data['area_name'].unique()), key='area_filter')
            selected_genders = st.sidebar.multiselect("Géneros", data['gender'].unique().tolist(),
                                                      default=list(data['gender'].unique()), key='gender_filter')
            selected_payments = st.sidebar.multiselect("Tipo de Pago", data['payment_type'].unique().tolist(),
                                                       default=list(data['payment_type'].unique()), key='payment_filter')
            selected_status = st.sidebar.multiselect("Estado", data['status'].unique().tolist(),
                                                     default=list(data['status'].unique()), key='status_filter')
            agg_level = st.sidebar.selectbox("Agrupar evolución por", ['Diario','Mensual','Anual'],
                                             index=1, key='agg_filter')
            df_f = data[
                (data['attention_date'] >= pd.to_datetime(start_date)) &
                (data['attention_date'] <= pd.to_datetime(end_date)) &
                (data['area_name'].isin(selected_areas)) &
                (data['gender'].isin(selected_genders)) &
                (data['payment_type'].isin(selected_payments)) &
                (data['status'].isin(selected_status))
            ]
            return df_f, agg_level

        df_f, agg_level = apply_filters(df)

        def plot_area_attention(data):
            fig, ax = plt.subplots(figsize=(8,5))
            order = data['area_name'].value_counts().index
            sns.countplot(data=data, y='area_name', order=order, ax=ax)
            ax.set_title("Atenciones por Área")
            ax.set_xlabel("Cantidad")
            ax.set_ylabel("Área")
            return fig

        def plot_top_treatments(data):
            fig, ax = plt.subplots(figsize=(8,5))
            top10 = data['treatment_name'].value_counts().nlargest(10)
            sns.barplot(x=top10.values, y=top10.index, ax=ax)
            ax.set_title("Top 10 Tratamientos")
            ax.set_xlabel("Cantidad")
            ax.set_ylabel("Tratamiento")
            return fig

        def plot_age_distribution(data):
            fig, ax = plt.subplots(figsize=(8,5))
            sns.histplot(data['age'], bins=20, kde=True, ax=ax)
            ax.set_title("Distribución de Edad de Pacientes")
            ax.set_xlabel("Edad")
            ax.set_ylabel("Frecuencia")
            return fig

        def plot_time_series(data, level):
            if level=='Diario':
                series = data.groupby('attention_date').size()
                xlabel='Fecha'
            elif level=='Mensual':
                series = data.groupby('year_month').size()
                xlabel='Mes'
            else:
                series = data.groupby('year').size()
                xlabel='Año'
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(series.index, series.values)
            ax.set_title(f"Evolución de Atenciones ({level})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Cantidad")
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
                index='area_name', columns='treatment_name', values='attention_id', aggfunc='count', fill_value=0
            )
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(pivot, cmap='YlGnBu', linewidths=.5, ax=ax)
            ax.set_title("Heatmap: Área vs Tratamiento")
            ax.set_xlabel("Tratamiento")
            ax.set_ylabel("Área")
            return fig

        # -------- Mostrar dashboard --------
        st.subheader("Atenciones por Área")
        st.pyplot(plot_area_attention(df_f))

        st.subheader("Top 10 Tratamientos")
        st.pyplot(plot_top_treatments(df_f))

        st.subheader("Distribución de Edad")
        st.pyplot(plot_age_distribution(df_f))

        st.subheader(f"Evolución {agg_level}")
        st.pyplot(plot_time_series(df_f, agg_level))

        st.subheader("Tipo de Pago y Estado de Pago")
        st.pyplot(plot_payment_status(df_f))

        st.subheader("Heatmap de Atención")
        st.pyplot(plot_heatmap(df_f))

        # -------- Dashboard de Predicción --------
        st.header('Dashboard de Predicción de Ingresos Mensuales')

        agg_pred = (df
                    .groupby(df['attention_date'].dt.to_period('M'))
                    .agg(ingreso_total=('price','sum'),
                         total_atenciones=('attention_id','count'),
                         pacientes_unicos=('patient_id', pd.Series.nunique),
                         pct_pagado=('status', lambda s: np.mean(s=='Pagado'))
                    )
                    .reset_index())
        agg_pred['fecha_mes'] = pd.to_datetime(agg_pred['attention_date'].dt.to_timestamp())
        agg_pred = agg_pred.sort_values('fecha_mes').reset_index(drop=True)

        agg_pred['sin_mes'] = np.sin(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
        agg_pred['cos_mes'] = np.cos(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
        agg_pred['anio_cent'] = agg_pred['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()
        agg_pred['lag1'] = agg_pred['ingreso_total'].shift(1).fillna(method='bfill')
        agg_pred['lag2'] = agg_pred['ingreso_total'].shift(2).fillna(method='bfill')
        agg_pred['ma3'] = agg_pred['ingreso_total'].rolling(3).mean().shift(1).fillna(method='bfill')

        variables = ['total_atenciones','pacientes_unicos','pct_pagado',
                     'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3']
        model = joblib.load(project_root/'models'/'models'/'modelo_final_2025_brutos.joblib')

        X = agg_pred[variables]
        y = agg_pred['ingreso_total']
        _, test_idx = train_test_split(agg_pred.index, test_size=0.2, random_state=42)
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        preds_test = model.predict(X_test)

        pred_df = pd.DataFrame({
            'fecha_mes': agg_pred.loc[test_idx,'fecha_mes'],
            'actual_income': y_test.values,
            'predicted_income': preds_test
        })

        st.subheader('Variables usadas por el modelo')
        for var in variables:
            st.write(f"- {var}")

        st.subheader('Evaluación en Test (parte de 2021–2024)')
        st.dataframe(pred_df.sort_values('fecha_mes'))
        col1, col2, col3 = st.columns(3)
        col1.metric('MAE', f"{mean_absolute_error(y_test, preds_test):,.2f}")
        col2.metric('RMSE', f"{np.sqrt(mean_squared_error(y_test, preds_test)):,.2f}")
        col3.metric('R²', f"{r2_score(y_test, preds_test):.3f}")

        st.sidebar.header('Predicción por Mes')
        selected_mes = st.sidebar.selectbox('Mes (1-12)', list(range(1,13)), index=0, key='pred_mes')
        if st.sidebar.button('Calcular', key='pred_button'):
            row = agg_pred[agg_pred['fecha_mes'].dt.month==selected_mes].iloc[0]
            X_in = pd.DataFrame({v:[row[v]] for v in variables})
            pred_point = model.predict(X_in)[0]
            st.write(f"**Predicción ingreso para mes {selected_mes}:** {pred_point:,.2f}")

        st.subheader('Histórico de Ingresos (2021–2024)')
        st.line_chart(agg_pred.set_index('fecha_mes')['ingreso_total'])

        future = pd.DataFrame({'fecha_mes': pd.date_range('2025-01-01','2025-12-01',freq='MS')})
        future['sin_mes'] = np.sin(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
        future['cos_mes'] = np.cos(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
        future['anio_cent'] = future['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()
        last = agg_pred.iloc[-1]
        for col in ['lag1','lag2','ma3']:
            future[col] = last[col]
        grp = agg_pred.groupby(agg_pred['fecha_mes'].dt.month)
        future['total_atenciones'] = future['fecha_mes'].dt.month.map(grp['total_atenciones'].mean())
        future['pacientes_unicos'] = future['fecha_mes'].dt.month.map(grp['pacientes_unicos'].mean())
        future['pct_pagado'] = future['fecha_mes'].dt.month.map(grp['pct_pagado'].mean())

        X_fut = future[variables]
        future['predicted_income'] = model.predict(X_fut)

        st.subheader('Pronóstico de Ingresos 2025')
        st.dataframe(future[['fecha_mes','predicted_income']].reset_index(drop=True))
        st.line_chart(future.set_index('fecha_mes')['predicted_income'])

# ===================== FIN PARTE 1/3 =====================

# ===================== PARTE 2/3 =====================

        # ========== GUARDAR TODO EN UN CSV Y MOSTRAR MODAL ==========
        EXPORT_DIR = Path(r"C:\Users\Perydox\Desktop\Proyecto_Grado - copia\models\reportes")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        csv_filename = f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = EXPORT_DIR / csv_filename

        # Incluye los datos principales del dashboard, las predicciones, y los agregados
        df_to_save = pd.DataFrame()
        with pd.ExcelWriter(csv_path.with_suffix('.xlsx')) as writer:
            df.to_excel(writer, sheet_name="Atenciones_Completas", index=False)
            pred_df.to_excel(writer, sheet_name="Prediccion_vs_Real", index=False)
            future[['fecha_mes', 'predicted_income']].to_excel(writer, sheet_name="Pronostico_2025", index=False)
        # También guarda un CSV plano solo con el pronóstico, por compatibilidad
        future[['fecha_mes', 'predicted_income']].to_csv(csv_path, index=False)

        st.success(f"Archivo generado: {csv_path.name} (y archivo Excel con todas las hojas)")

        # Almacenar info para el CRUD
        st.session_state['crud_data'] = {
            'usuario': st.session_state.get('username', ''),
            'archivo_csv': str(csv_path),
            'archivo_excel': str(csv_path.with_suffix('.xlsx')),
            'parametros': json.dumps({
                'generado_por': st.session_state.get('username', ''),
                'fecha': datetime.now().isoformat()
            }),
            'fecha_creacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.session_state['show_crud'] = True

        # ========== FORMULARIO CRUD EN MODAL ==========
    if st.session_state.get('show_crud', False):
        st.markdown("### Registrar reporte en la base de datos")
    # Puedes usar st.expander para dar efecto de "modal/plegable":
    with st.expander("Completa los datos del reporte y guarda en la base de datos", expanded=True):
        crud_data = st.session_state.get('crud_data', {})
        usuario = crud_data.get('usuario', '')
        archivo_csv = crud_data.get('archivo_csv', '')
        archivo_excel = crud_data.get('archivo_excel', '')
        parametros = crud_data.get('parametros', '')
        fecha_creacion = crud_data.get('fecha_creacion', '')

        # Vista previa CSV
        st.markdown("**Vista previa del archivo CSV (Pronóstico):**")
        if archivo_csv and Path(archivo_csv).exists():
            try:
                preview_df = pd.read_csv(archivo_csv)
                st.dataframe(preview_df.head(20))
            except Exception as e:
                st.error(f"No se puede mostrar la vista previa del CSV: {e}")

        with st.form("form_prediccion"):
            nombre = st.text_input("Nombre del reporte", value=f"Reporte {fecha_creacion[:10]}")
            tipo = st.selectbox("Tipo", options=["Gráfica", "Predicción", "Ambos"], index=2)
            descripcion = st.text_area("Descripción")
            submit_crud = st.form_submit_button("Guardar registro en base de datos")

        if submit_crud:
            st.session_state['crud_submit'] = {
                'usuario': usuario,
                'fecha_creacion': fecha_creacion,
                'nombre': nombre,
                'tipo': tipo,
                'archivo_csv': archivo_csv,
                'parametros': parametros,
                'descripcion': descripcion
            }
        else:
            st.session_state['crud_submit'] = None
 

# ===================== FIN PARTE 2/3 =====================
# ===================== PARTE 3/3 =====================

# Guardar el registro en la tabla predicciones si el formulario fue enviado
if st.session_state.get('crud_submit') is not None:
    db_path = project_root / "data" / "consultorio.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    data = st.session_state['crud_submit']

    # Ajusta la consulta SQL para incluir la descripción
    cursor.execute("""
        INSERT INTO predicciones (usuario, fecha_creacion, nombre, tipo, archivo_csv, parametros, descripcion)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        data['usuario'],
        data['fecha_creacion'],
        data['nombre'],
        data['tipo'],
        data['archivo_csv'],
        data['parametros'],
        data['descripcion']
    ))
    conn.commit()
    conn.close()

    st.success("¡Registro guardado exitosamente en la base de datos!")
    st.session_state['crud_submit'] = None
    st.session_state['show_crud'] = False

# ===================== FIN PARTE 3/3 =====================
# ============ sistema_web3.py — PARTE 4 / 4 ============
# (Añade este bloque tras la PARTE 3)

# ---------- Pestaña 2 » Predicción & Reporte ----------
with tabs[2]:
    if not st.session_state.data_loaded:
        st.info("Carga primero los datos para generar predicciones.")
        st.stop()

    df = st.session_state.df
    st.subheader("Predicción de ingresos mensuales")

    # --- Feature engineering compacto ---
    agg = (
        df.groupby(df["attention_date"].dt.to_period("M"))
          .agg(ingreso=("price", "sum"),
               atenciones=("attention_id", "count"),
               pacientes=("patient_id", pd.Series.nunique),
               pct_pagado=("status", lambda s: np.mean(s == "Pagado")))
          .reset_index()
    )
    agg["fecha"] = pd.to_datetime(agg["attention_date"].dt.to_timestamp()).sort_values()
    agg.sort_values("fecha", inplace=True)

    agg["sin_mes"]  = np.sin(2 * np.pi * (agg["fecha"].dt.month - 1) / 12)
    agg["cos_mes"]  = np.cos(2 * np.pi * (agg["fecha"].dt.month - 1) / 12)
    agg["anio_cent"] = agg["fecha"].dt.year - agg["fecha"].dt.year.min()
    agg["lag1"] = agg["ingreso"].shift(1).bfill()
    agg["lag2"] = agg["ingreso"].shift(2).bfill()
    agg["ma3"]  = agg["ingreso"].rolling(3).mean().shift(1).bfill()

    FEAT = ["atenciones", "pacientes", "pct_pagado",
            "sin_mes", "cos_mes", "anio_cent", "lag1", "lag2", "ma3"]

    model = joblib.load(MODELS_DIR / "modelo_final_2025_brutos.joblib")

    # --- Evaluación rápida ---
    _, tst_idx = train_test_split(agg.index, test_size=0.2, random_state=42)
    y_test = agg.loc[tst_idx, "ingreso"]
    y_pred = model.predict(agg.loc[tst_idx, FEAT])

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",  f"{mean_absolute_error(y_test, y_pred):,.0f}")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
    c3.metric("R²",   f"{r2_score(y_test, y_pred):.3f}")

    # --- Predicción puntual ---
    st.sidebar.header("Predicción puntual")
    mes_sel = st.sidebar.selectbox("Mes (1‑12)", list(range(1, 13)))
    if st.sidebar.button("Calcular"):
        fila = agg[agg["fecha"].dt.month == mes_sel].iloc[0][FEAT]
        valor = float(model.predict([fila]))
        st.success(f"Ingreso estimado para mes {mes_sel}: **{valor:,.0f}**")

    # --- Histórico ---
    st.line_chart(agg.set_index("fecha")["ingreso"])

    # --- Pronóstico 2025 (12 meses) ---
    fut = pd.DataFrame({"fecha": pd.date_range("2025‑01‑01", periods=12, freq="MS")})
    fut["sin_mes"] = np.sin(2 * np.pi * (fut["fecha"].dt.month - 1) / 12)
    fut["cos_mes"] = np.cos(2 * np.pi * (fut["fecha"].dt.month - 1) / 12)
    fut["anio_cent"] = fut["fecha"].dt.year - agg["fecha"].dt.year.min()

    last = agg.iloc[-1]
    for col in ["lag1", "lag2", "ma3"]:
        fut[col] = last[col]

    grp = agg.groupby(agg["fecha"].dt.month)
    fut["atenciones"] = fut["fecha"].dt.month.map(grp["atenciones"].mean())
    fut["pacientes"]  = fut["fecha"].dt.month.map(grp["pacientes"].mean())
    fut["pct_pagado"] = fut["fecha"].dt.month.map(grp["pct_pagado"].mean())

    fut["pred"] = model.predict(fut[FEAT])

    st.subheader("Pronóstico 2025")
    st.line_chart(fut.set_index("fecha")["pred"])

    # --- Exportar reporte ---
    if st.button("📤 Exportar reporte"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path   = REPORTS_DIR / f"pronostico_{stamp}.csv"
        excel_path = csv_path.with_suffix(".xlsx")

        fut[["fecha", "pred"]].to_csv(csv_path, index=False)
        with pd.ExcelWriter(excel_path) as xl:
            df.to_excel(xl, sheet_name="Atenciones", index=False)
            agg.to_excel(xl, sheet_name="Histórico",  index=False)
            fut.to_excel(xl, sheet_name="Pronóstico", index=False)

        st.toast("Reporte exportado ✔️", icon="📄")

        # Guardamos datos para CRUD
        st.session_state.crud = {
            "usuario": st.session_state.username,
            "csv": str(csv_path),
            "excel": str(excel_path),
            "fecha": datetime.now().strftime("%Y‑%m‑%d %H:%M:%S")
        }
        st.session_state.show_crud = True

    # --- Formulario CRUD minimal ---
    if st.session_state.get("show_crud", False):
        with st.expander("Registrar reporte en base de datos", expanded=True):
            cd = st.session_state.crud
            st.dataframe(pd.read_csv(cd["csv"]).head(), use_container_width=True)

            with st.form("crud_form"):
                nombre = st.text_input("Nombre del reporte", value=f"Reporte {cd['fecha'][:10]}")
                tipo   = st.selectbox("Tipo", ["Predicción", "Gráfica", "Ambos"], index=0)
                descripcion = st.text_area("Descripción")
                if st.form_submit_button("Guardar"):
                    db = DATA_DIR / "consultorio.db"
                    conn = sqlite3.connect(db)
                    cur  = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO predicciones
                        (usuario, fecha_creacion, nombre, tipo, archivo_csv, parametros, descripcion)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cd["usuario"],
                            cd["fecha"],
                            nombre,
                            tipo,
                            cd["csv"],
                            json.dumps({"excel": cd["excel"]}),
                            descripcion,
                        ),
                    )
                    conn.commit(); conn.close()
                    st.toast("Registro guardado ✔️", icon="💾")
                    st.session_state.show_crud = False
