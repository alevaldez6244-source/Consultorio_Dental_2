# Archivo: app/views/login_view.py
import streamlit as st
import base64
from pathlib import Path
from app.views.styles import load_css

class LoginView:
    def render(self):
        """Dibuja la pantalla de login."""
        load_css() # Aplicar estilos específicos
        
        _, col_center, _ = st.columns([2, 3, 2])
        with col_center:
            self._render_logo()
            
            with st.form("login_form"):
                st.markdown("<h3 style='text-align:center;'>Iniciar sesión</h3>", unsafe_allow_html=True)
                usuario = st.text_input("Usuario")
                password = st.text_input("Contraseña", type="password")
                submit = st.form_submit_button("Entrar")
                
                return submit, usuario, password

    def _render_logo(self):
        """Muestra el logo circular."""
        logo_path = Path("app/assets/logo.png") # Ajusta la ruta si es necesario
        if logo_path.exists():
            encoded = base64.b64encode(logo_path.read_bytes()).decode()
            st.markdown(
                f"""<div style="display:flex;justify-content:center;margin-bottom:1rem;">
                    <img src="data:image/png;base64,{encoded}" 
                    style="width:150px;height:150px;border-radius:50%;object-fit:cover;"/>
                </div>""", 
                unsafe_allow_html=True
            )