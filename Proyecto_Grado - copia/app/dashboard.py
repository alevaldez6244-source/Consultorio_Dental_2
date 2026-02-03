
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
@st.cache_data
def load_data():
    areas = pd.read_csv(r"C:\Users\Perydox\Desktop\Proyecto_Grado - copia\data\Datos_final\areas_final.csv",
                        header=None, names=['area_id', 'area_name'])
    pacientes = pd.read_csv(r"C:\Users\Perydox\Desktop\Proyecto_Grado - copia\data\Datos_final\pacientes_final.csv",
                            header=None, names=['patient_id', 'patient_name', 'age', 'gender', 'registration_date'])
    tratamientos = pd.read_csv(r"C:\Users\Perydox\Desktop\Proyecto_Grado - copia\data\Datos_final\tratamientos_final.csv",
                               header=None, names=['treatment_id', 'treatment_name', 'area_id', 'price'])
    atenciones = pd.read_csv(r"C:\Users\Perydox\Desktop\Proyecto_Grado - copia\data\Datos_final\atenciones_final.csv",
                             header=None, names=['attention_id', 'patient_id', 'treatment_id', 'attention_date', 'payment_type', 'status', 'extra'])
    atenciones.drop(columns=['extra'], inplace=True)

    df = (
        atenciones
        .merge(pacientes, on='patient_id', how='left')
        .merge(tratamientos, on='treatment_id', how='left')
        .merge(areas, on='area_id', how='left')
    )
    df['attention_date'] = pd.to_datetime(df['attention_date']) 
    df['year_month'] = df['attention_date'].dt.to_period('M').dt.to_timestamp()
    df['year'] = df['attention_date'].dt.year
    return df


st.set_page_config(page_title="Dashboard Dental", layout="wide")
st.title("Dashboard de Atenciones Dentales Personalizado")
df = load_data()
sns.set_style("whitegrid")

# 4. Sidebar: filtros de fecha, áreas y más
st.sidebar.header("Filtros")
default_min, default_max = df['attention_date'].min(), df['attention_date'].max()
start_date, end_date = st.sidebar.date_input(
    "Rango de fechas",
    [default_min, default_max],
    min_value=default_min,
    max_value=default_max
)
area_options = df['area_name'].unique().tolist()
gender_options = df['gender'].unique().tolist()
payment_options = df['payment_type'].unique().tolist()
status_options = df['status'].unique().tolist()

selected_areas = st.sidebar.multiselect("Áreas", area_options, default=area_options)
selected_genders = st.sidebar.multiselect("Géneros", gender_options, default=gender_options)
selected_payments = st.sidebar.multiselect("Tipo de Pago", payment_options, default=payment_options)
selected_status = st.sidebar.multiselect("Estado", status_options, default=status_options)
agg_level = st.sidebar.selectbox("Agrupar evolución por", ['Diario', 'Mensual', 'Anual'], index=1)
df_f = df[
    (df['attention_date'] >= pd.to_datetime(start_date)) &
    (df['attention_date'] <= pd.to_datetime(end_date)) &
    (df['area_name'].isin(selected_areas)) &
    (df['gender'].isin(selected_genders)) &
    (df['payment_type'].isin(selected_payments)) &
    (df['status'].isin(selected_status))
]

# 6. Funciones de plot
def plot_area_attention(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = data['area_name'].value_counts().index
    sns.countplot(data=data, y='area_name', order=order, ax=ax)
    ax.set_title("Atenciones por Área")
    ax.set_xlabel("Cantidad")
    ax.set_ylabel("Área")
    return fig
def plot_top_treatments(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    top10 = data['treatment_name'].value_counts().nlargest(10)
    sns.barplot(x=top10.values, y=top10.index, ax=ax)
    ax.set_title("Top 10 Tratamientos")
    ax.set_xlabel("Cantidad")
    ax.set_ylabel("Tratamiento")
    return fig
def plot_age_distribution(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data['age'], bins=20, kde=True, ax=ax)
    ax.set_title("Distribución de Edad de Pacientes")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Frecuencia")
    return fig
def plot_time_series(data, level):
    if level == 'Diario':
        series = data.groupby('attention_date').size()
        xlabel = 'Fecha'
    elif level == 'Mensual':
        series = data.groupby('year_month').size()
        xlabel = 'Mes'
    else:
        series = data.groupby('year').size()
        xlabel = 'Año'
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values)
    ax.set_title(f"Evolución de Atenciones ({level})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cantidad")
    return fig

def plot_payment_status(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    data['payment_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
    ax1.set_title("Tipo de Pago")
    ax1.set_ylabel("")
    data['status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2)
    ax2.set_title("Estado de Pago")
    ax2.set_ylabel("")
    return fig

def plot_heatmap(data):
    pivot = data.pivot_table(
        index='area_name',
        columns='treatment_name',
        values='attention_id',
        aggfunc='count',
        fill_value=0
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=.5, ax=ax)
    ax.set_title("Heatmap: Área vs Tratamiento")
    ax.set_xlabel("Tratamiento")
    ax.set_ylabel("Área")
    return fig

# 7. Render de gráficos
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

# 8. Punto de entrada
def main():
    pass

if __name__ == "__main__":
    main()