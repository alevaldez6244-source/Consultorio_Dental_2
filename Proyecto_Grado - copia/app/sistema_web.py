# Archivo: sistema_web.py
import sys
from pathlib import Path
import time

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

# Configuración de página
st.set_page_config(page_title="Consultorio Dental - Sistema Web", layout="wide")

# === LOGIN ===
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

# === CONTROL DE ANIMACIÓN Y CARGA SOLO UNA VEZ ===
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

def reset_data_loaded():
    st.session_state.data_loaded = False
    st.session_state.df = None

uploader_placeholder = st.empty()
dashboard_placeholder = st.empty()

# ===================== BOTÓN DE VOLVER A INGRESAR DATOS ======================
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

# SOLO animar al cargar archivos nuevos y no recargar por filtros
if all([areas_file, pacientes_file, tratamientos_file, atenciones_file]) and not st.session_state.data_loaded:
    with st.spinner("Procesando archivos, por favor espera... (10 s)"):
        progress = st.progress(0)
        for i in range(10):
            time.sleep(1)
            progress.progress((i + 1) / 10)
    # --- Cargar solo una vez y guardar en session_state
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

# SOLO usa el DataFrame cacheado, nunca recargues
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    with dashboard_placeholder.container():
        st.sidebar.title(f"Usuario: {st.session_state.username}")

        # --- Descriptivo ---
        st.title("Dashboard de Atenciones Dentales Personalizado")
        sns.set_style("whitegrid")

        # Filtros
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
        # ======================== GRÁFICOS =========================

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

        # ======================== DASHBOARD VISUALIZACIÓN =========================

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

        # ===================== DASHBOARD DE PREDICCIÓN ======================

        st.header('Dashboard de Predicción de Ingresos Mensuales')

        # Preparar datos agregados
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

        # Crear features
        agg_pred['sin_mes'] = np.sin(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
        agg_pred['cos_mes'] = np.cos(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
        agg_pred['anio_cent'] = agg_pred['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()
        agg_pred['lag1'] = agg_pred['ingreso_total'].shift(1).fillna(method='bfill')
        agg_pred['lag2'] = agg_pred['ingreso_total'].shift(2).fillna(method='bfill')
        agg_pred['ma3'] = agg_pred['ingreso_total'].rolling(3).mean().shift(1).fillna(method='bfill')

        # Variables y modelo
        variables = ['total_atenciones','pacientes_unicos','pct_pagado',
                     'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3']
        model = joblib.load(project_root/'models'/'models'/'modelo_final_2025_brutos.joblib')

        # Evaluación en test
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

        # Mostrar variables usadas
        st.subheader('Variables usadas por el modelo')
        for var in variables:
            st.write(f"- {var}")

        # Evaluación en Test (2021–2024)
        st.subheader('Evaluación en Test (parte de 2021–2024)')
        st.dataframe(pred_df.sort_values('fecha_mes'))
        col1, col2, col3 = st.columns(3)
        col1.metric('MAE', f"{mean_absolute_error(y_test, preds_test):,.2f}")
        col2.metric('RMSE', f"{np.sqrt(mean_squared_error(y_test, preds_test)):,.2f}")
        col3.metric('R²', f"{r2_score(y_test, preds_test):.3f}")

        # Predicción puntual por mes
        st.sidebar.header('Predicción por Mes')
        selected_mes = st.sidebar.selectbox('Mes (1-12)', list(range(1,13)), index=0, key='pred_mes')
        if st.sidebar.button('Calcular', key='pred_button'):
            row = agg_pred[agg_pred['fecha_mes'].dt.month==selected_mes].iloc[0]
            X_in = pd.DataFrame({v:[row[v]] for v in variables})
            pred_point = model.predict(X_in)[0]
            st.write(f"**Predicción ingreso para mes {selected_mes}:** {pred_point:,.2f}")

        # Histórico de Ingresos
        st.subheader('Histórico de Ingresos (2021–2024)')
        st.line_chart(agg_pred.set_index('fecha_mes')['ingreso_total'])

        # Pronóstico 2025
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
