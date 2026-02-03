# Archivo: app/utils/helpers.py
import streamlit as st

def toast(msg: str, icon: str = ""):
    """Muestra una notificación flotante (Toast) o un mensaje de éxito."""
    try:
        st.toast(msg, icon=icon)
    except AttributeError:
        st.success(msg)