# Archivo: app/views/styles.py
import streamlit as st

def load_css():
    """Carga los estilos CSS personalizados de la aplicación."""
    st.markdown("""
        <style>
        /* === 1. GENERAL === */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* === 2. SIDEBAR PERSONALIZADO (Opción 3) === */
        
        /* Ocultar los círculos de radio por defecto */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
            display: None;
        }

        /* Contenedor de las opciones */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            gap: 8px;
        }

        /* Botones del menú (Estado inactivo) */
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            background-color: transparent;
            padding: 10px 20px;
            border-radius: 5px;
            border-left: 4px solid transparent;
            color: #cfcfcf;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 2px;
        }

        /* Efecto Hover */
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background-color: rgba(255, 255, 255, 0.05);
            padding-left: 25px;
            color: #ffffff;
        }

        /* ESTADO ACTIVO (CORREGIDO) */
        /* Usamos :checked en lugar de [checked] para detectar el cambio de estado real */
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(76, 175, 80, 0.15) 0%, transparent 100%);
            border-left: 4px solid #4CAF50;
            color: #4CAF50 !important;
            font-weight: 600;
        }

        /* === 3. ESTILOS DE LOGIN (MANTENIDOS) === */
        [data-testid="stForm"] {
            max-width: 320px;
            margin: 0 auto;
            padding: 1.5rem 2rem 2rem;
            border: 1px solid #3a3a3a;
            border-radius: 0.75rem;
            background: #1e1e1e;
            box-shadow: 0 4px 12px rgba(0,0,0,0.45);
        }
        
        [data-testid="stForm"] input {
            padding: 0.4rem 0.6rem !important;
        }

        /* === 4. MÉTRICAS / KPIs (MEJORADO) === */
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #464b5c;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #4CAF50;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #4CAF50 !important;
        }
        </style>
    """, unsafe_allow_html=True)