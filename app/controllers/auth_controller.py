# Archivo: app/controllers/auth_controller.py
import streamlit as st
from app.auth import check_credentials  # Usamos tu archivo auth.py original
from app.views.login_view import LoginView

class AuthController:
    def __init__(self):
        self.view = LoginView()

    def manage_auth(self):
        """
        Gestiona el flujo de login.
        Retorna True si el usuario está autenticado, False si no.
        """
        # 1. Verificar si ya hay sesión activa
        if st.session_state.get("authenticated", False):
            return True

        # 2. Si no, mostrar formulario de Login
        submit, user, password = self.view.render()

        if submit:
            if check_credentials(user, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = user
                st.rerun()  # Recargar para entrar a la app
            else:
                st.error("❌ Usuario o contraseña incorrectos")
        
        return False

    def logout(self):
        """Cierra la sesión y limpia el estado."""
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.session_state["df"] = None
        st.session_state["data_loaded"] = False
        st.rerun()