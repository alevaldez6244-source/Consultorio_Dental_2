# Archivo: app/views/login_view.py
import streamlit as st
import base64
from pathlib import Path
from app.views.styles import load_css

class LoginView:
    def render(self):
        """Dibuja la pantalla de login."""
        load_css() # Aplicar estilos
        
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
        """Muestra el logo circular usando RUTA ABSOLUTA DINÁMICA."""
        
        # 1. Obtenemos la ruta absoluta de ESTE archivo (login_view.py)
        # Ejemplo: C:\Users\Perydox\...\app\views\login_view.py
        current_file = Path(__file__).resolve()
        
        # 2. Navegamos hacia la carpeta 'assets'
        # .parent = app/views
        # .parent.parent = app
        # / "assets" / "logo.png" = app/assets/logo.png
        logo_path = current_file.parent.parent / "assets" / "logo.png"
        
        if logo_path.exists():
            encoded = base64.b64encode(logo_path.read_bytes()).decode()
            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; margin-bottom:1rem;">
                    <img src="data:image/png;base64,{encoded}" 
                    style="
                        width: 150px; 
                        height: 150px; 
                        border-radius: 50%; 
                        object-fit: cover; 
                        border: 3px solid #4CAF50;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    "/>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            # Si falla, mostramos la ruta exacta que está intentando buscar para depurar
            st.error(f"❌ No se encuentra el logo en:\n{logo_path}")