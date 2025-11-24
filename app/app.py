import sys
from pathlib import Path

import streamlit as st
from modules.data_loader   import load_data
from modules.visualization import (plot_area_attention,   # + otras funciones
                                   )
from modules.predictor     import IncomePredictor

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "modelo_final.joblib"

# 1. Login (igual que antes) ── …

# 2. Carga de CSV
st.header("Carga de datos CSV")
areas_f        = st.file_uploader("Áreas",        type="csv")
pacientes_f    = st.file_uploader("Pacientes",    type="csv")
tratamientos_f = st.file_uploader("Tratamientos", type="csv")
atenciones_f   = st.file_uploader("Atenciones",   type="csv")

if all([areas_f, pacientes_f, tratamientos_f, atenciones_f]):
    df = load_data(areas_f, pacientes_f, tratamientos_f, atenciones_f)

    # 3. Análisis Descriptivo
    st.subheader("Atenciones por Área")
    st.pyplot(plot_area_attention(df))
    # … (resto de visualizaciones)

    # 4. Predicción
    pred = IncomePredictor(MODEL_PATH)
    # a) batch
    df_monthly = (
        df.groupby(df['attention_date'].dt.to_period('M'))
          .agg(ingreso_total=('price','sum'))
          .reset_index()
          .rename(columns={'attention_date':'fecha_mes'})
    )
    df_monthly['fecha_mes'] = pd.to_datetime(df_monthly['fecha_mes'].dt.to_timestamp())
    df_monthly['predicted_income'] = pred.predict_batch(df_monthly)

    st.subheader("Histórico de Ingresos y Pronóstico")
    st.line_chart(df_monthly.set_index('fecha_mes')[['ingreso_total',
                                                     'predicted_income']])

    # b) predicción puntual
    mes = st.sidebar.number_input("Mes 1‑12", 1, 12, 1)
    row = df_monthly[df_monthly['fecha_mes'].dt.month == mes].iloc[0:1]
    st.sidebar.write(f"Predicción: {pred.predict_one(row):,.2f}")
