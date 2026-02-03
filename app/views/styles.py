# Archivo: app/views/styles.py
import streamlit as st

def load_css():
    """Carga los estilos CSS personalizados de la aplicación."""
    st.markdown("""
        <style>
        /* Formulario de Login Compacto */
        [data-testid="stForm"] {
            max-width: 320px;
            margin: 0 auto;
            padding: 1.5rem 2rem 2rem;
            border: 1px solid #3a3a3a;
            border-radius: 0.75rem;
            background: #1e1e1e;
            box-shadow: 0 4px 12px rgba(0,0,0,0.45);
        }
        
        /* Inputs más pequeños en formularios */
        [data-testid="stForm"] input {
            padding: 0.4rem 0.6rem !important;
        }

        /* Estilo para las métricas (KPIs) */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)