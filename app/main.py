import streamlit as st
import sys
import pandas as pd
from pathlib import Path

# --- 1. SETUP DE RUTAS (CRÍTICO) ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="🦷 Consultorio Dental Inteligente",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. IMPORTACIONES CONTROLADAS ---
try:
    import app.config as config
    from app.controllers.auth_controller import AuthController
    from app.controllers.app_controller import AppController
except ImportError as e:
    st.error(f"❌ Error Crítico de Importación: {e}")
    st.stop()

def main():
    # GESTIÓN DE SEGURIDAD
    auth = AuthController()
    if not auth.manage_auth():
        return

    # INICIALIZACIÓN DEL CONTROLADOR
    controller = AppController()

    # SIDEBAR
    with st.sidebar:
        try:
            logo_path = Path(__file__).parent / "assets" / "logo.png"
            if logo_path.exists():
                st.image(str(logo_path), width=120)
            else:
                st.header("🦷 Consultorio")
        except Exception:
            st.header("🦷 Consultorio")

        st.markdown(f"Hola, **{st.session_state.get('username', 'Doctor')}**")
        st.markdown("---")

        opcion_menu = st.radio(
            "Navegación",
            [
                "🏠 Inicio", 
                "📥 Carga de Datos", 
                "📊 Dashboard", 
                "🔮 Predicción IA", 
                "♟️ Estrategia"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button("Cerrar Sesión", type="secondary"):
            auth.logout()

    # ENRUTADOR DE VISTAS
    if opcion_menu == "🏠 Inicio":
        st.subheader("Información general del consultorio")

        # 1. Datos Fijos (Solicitados)
        areas_data = [
            {"ID": 1, "Área": "Odontología General"},
            {"ID": 2, "Área": "Ortodoncia"},
            {"ID": 3, "Área": "Endodoncia"},
            {"ID": 4, "Área": "Periodoncia"},
            {"ID": 5, "Área": "Odontopediatría"},
            {"ID": 6, "Área": "Prostodoncia"},
            {"ID": 7, "Área": "Cirugía Oral"},
        ]
        
        tratamientos_data = [
            {"Tratamiento": "Consulta General", "Área": "Odontología General", "Costo": 60},
            {"Tratamiento": "Limpieza Dental", "Área": "Odontología General", "Costo": 180},
            {"Tratamiento": "Tratamiento de Caries", "Área": "Odontología General", "Costo": 200},
            {"Tratamiento": "Colocación de Brackets", "Área": "Ortodoncia", "Costo": 3000},
            {"Tratamiento": "Mantenimiento de Ortodoncia", "Área": "Ortodoncia", "Costo": 200},
            {"Tratamiento": "Tratamiento de Conducto", "Área": "Endodoncia", "Costo": 150},
            {"Tratamiento": "Retratamiento de Conducto", "Área": "Endodoncia", "Costo": 300},
            {"Tratamiento": "Tratamiento de Encías", "Área": "Periodoncia", "Costo": 200},
            {"Tratamiento": "Cirugía de Encías", "Área": "Periodoncia", "Costo": 200},
            {"Tratamiento": "Consulta Infantil", "Área": "Odontopediatría", "Costo": 60},
            {"Tratamiento": "Aplicación de Flúor", "Área": "Odontopediatría", "Costo": 20},
            {"Tratamiento": "Prótesis Dental Parcial", "Área": "Prostodoncia", "Costo": 800},
            {"Tratamiento": "Prótesis Completa", "Área": "Prostodoncia", "Costo": 600},
            {"Tratamiento": "Corona de Oro", "Área": "Prostodoncia", "Costo": 2000},
            {"Tratamiento": "Corona de Cromo", "Área": "Prostodoncia", "Costo": 250},
            {"Tratamiento": "Corona Jacket", "Área": "Prostodoncia", "Costo": 200},
            {"Tratamiento": "Perno Muñón", "Área": "Prostodoncia", "Costo": 120},
            {"Tratamiento": "Prótesis Fija Canino a Canino (pieza de cromo)", "Área": "Prostodoncia", "Costo": 1400},
            {"Tratamiento": "Extracción de Muelas del Juicio", "Área": "Cirugía Oral", "Costo": 500},
        ]

        df_areas = pd.DataFrame(areas_data)
        df_trat = pd.DataFrame(tratamientos_data)

        # 2. Métricas Generales
        c1, c2, c3 = st.columns(3)
        c1.metric("Áreas totales", len(df_areas))
        c2.metric("Tratamientos totales", len(df_trat))
        c3.metric("Rango de costos", f"{df_trat['Costo'].min()} – {df_trat['Costo'].max():,.0f}")

        # 3. Tablas Visuales
        st.markdown("### Áreas")
        st.table(df_areas)

        st.markdown("### Tratamientos y costos")
        st.table(df_trat)

    elif opcion_menu == "📥 Carga de Datos":
        controller.run_uploader()

    elif opcion_menu == "📊 Dashboard":
        controller.run_dashboard()

    elif opcion_menu == "🔮 Predicción IA":
        controller.run_prediction()

    elif opcion_menu == "♟️ Estrategia":
        controller.run_strategy()

if __name__ == "__main__":
    main()