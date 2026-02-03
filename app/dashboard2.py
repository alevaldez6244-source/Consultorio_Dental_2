import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'Datos_final'
MODEL_PATH = Path(r"C:\Users\Perydox\Desktop\Proyecto_Grado - copia\models\models\modelo_final_2025_brutos.joblib")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
@st.cache_data
def load_data():
    BASE = DATA_DIR
    areas = pd.read_csv(BASE / 'areas_final.csv', header=None, names=['area_id','area_name'])
    pacientes = pd.read_csv(BASE / 'pacientes_final.csv', header=None,
                            names=['patient_id','patient_name','age','gender','registration_date'])
    tratamientos = pd.read_csv(BASE / 'tratamientos_final.csv', header=None,
                               names=['treatment_id','treatment_name','area_id','price'])
    atenciones = pd.read_csv(BASE / 'atenciones_final.csv', header=None,
                              names=['attention_id','patient_id','treatment_id','attention_date','payment_type','status','extra'])
    atenciones = atenciones.drop(columns=['extra'])
    df = (
        atenciones
        .merge(pacientes, on='patient_id', how='left')
        .merge(tratamientos, on='treatment_id', how='left')
        .merge(areas, on='area_id', how='left')
    )
    df['attention_date'] = pd.to_datetime(df['attention_date'])
    df['anio'] = df['attention_date'].dt.year
    df['mes'] = df['attention_date'].dt.month
    agg = df.groupby(['anio','mes']).agg(
        ingreso_total=('price','sum'),
        total_atenciones=('attention_id','count'),
        pacientes_unicos=('patient_id', pd.Series.nunique),
        pct_pagado=('status', lambda s: np.mean(s=='Pagado'))
    ).reset_index()
    agg['fecha_mes'] = pd.to_datetime(
        agg['anio'].astype(str) + '-' + agg['mes'].astype(str).str.zfill(2) + '-01'
    )
    agg = agg.sort_values('fecha_mes').reset_index(drop=True)
    agg['sin_mes'] = np.sin(2*np.pi*(agg['mes']-1)/12)
    agg['cos_mes'] = np.cos(2*np.pi*(agg['mes']-1)/12)
    agg['anio_cent'] = agg['anio'] - agg['anio'].min()
    agg['lag1'] = agg['ingreso_total'].shift(1).fillna(method='bfill')
    agg['lag2'] = agg['ingreso_total'].shift(2).fillna(method='bfill')
    agg['ma3'] = agg['ingreso_total'].rolling(3).mean().shift(1).fillna(method='bfill')
    return agg

data = load_data()
model = load_model()
variables = [
    'total_atenciones','pacientes_unicos','pct_pagado',
    'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3'
]

# Preparar matriz de features
df_feat = data[variables]

# Separar un test interno para evaluación
_, test_idx = train_test_split(data.index, test_size=0.2, random_state=42, shuffle=True)
X_test = df_feat.loc[test_idx]
y_test = data.loc[test_idx, 'ingreso_total']
dates_test = data.loc[test_idx, 'fecha_mes']

# Predicciones en test
preds_test = model.predict(X_test)
pred_df = pd.DataFrame({
    'fecha_mes': dates_test.values,
    'actual_income': y_test.values,
    'predicted_income': preds_test
}).sort_values('fecha_mes').reset_index(drop=True)

# Cálculo de métricas en test
mae_test = mean_absolute_error(y_test, preds_test)
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
r2_test = r2_score(y_test, preds_test)

# --- Streamlit UI ---
st.title('Dashboard de Predicción de Ingresos Mensuales')
st.subheader('Variables usadas por el modelo')
for var in variables:
    st.write(f"- {var}")


st.subheader('Evaluación en Test (parte de 2021–2024)')
st.dataframe(pred_df)
st.subheader('Métricas en Test')
col1, col2, col3 = st.columns(3)
col1.metric('MAE', f"{mae_test:,.2f}")
col2.metric('RMSE', f"{rmse_test:,.2f}")
col3.metric('R²', f"{r2_test:.3f}")

# Predicción puntual por mes
st.sidebar.header('Predicción por Mes')
selected_mes = st.sidebar.selectbox('Mes (1-12)', list(range(1,13)), index=0)
if st.sidebar.button('Calcular'):
    hist = data[data['mes']==selected_mes].iloc[0]
    X_in = pd.DataFrame({v:[hist[v]] for v in variables})
    pred = model.predict(X_in)[0]
    st.write(f"**Predicción ingreso para mes {selected_mes}:** {pred:,.2f}")

st.subheader('Histórico de Ingresos (2021–2024)')
st.line_chart(data.set_index('fecha_mes')['ingreso_total'])
future = pd.DataFrame({'fecha_mes': pd.date_range('2025-01-01','2025-12-01',freq='MS')})
future['anio'] = future['fecha_mes'].dt.year
future['mes'] = future['fecha_mes'].dt.month
for v in ['total_atenciones','pacientes_unicos','pct_pagado']:
    future[v] = future['mes'].map(data.groupby('mes')[v].mean())
future['sin_mes'] = np.sin(2*np.pi*(future['mes']-1)/12)
future['cos_mes'] = np.cos(2*np.pi*(future['mes']-1)/12)
future['anio_cent'] = future['anio'] - data['anio'].min() 
for lag in ['lag1','lag2','ma3']:
    future[lag] = data[lag].iloc[-1]
X_fut = future[variables]
future['predicted_income'] = model.predict(X_fut)
st.subheader('Pronóstico de Ingresos 2025')
st.dataframe(future[['fecha_mes','predicted_income']].reset_index(drop=True))
st.line_chart(future.set_index('fecha_mes')['predicted_income'])
st.markdown('---')
st.write('Panel interactivo para históricos, evaluación en test y pronóstico de ingresos.')

