import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Importaciones MVC
from app.models.data_service import DataService
from app.models.ml_service import MLService
from app.models.db_service import DBService
from app.views.upload_view import UploadView
from app.views.dashboard_view import DashboardView
from app.views.prediction_view import PredictionView
from app.views.strategy_view import StrategyView
from app.utils.helpers import toast

class AppController:
    def __init__(self):
        # Servicios
        self.data_service = DataService()
        self.ml_service = MLService()
        self.db_service = DBService()
        
        # Vistas
        self.upload_view = UploadView()
        self.dashboard_view = DashboardView()
        self.pred_view = PredictionView()
        self.strategy_view = StrategyView()

        # Estado Global
        if "df" not in st.session_state:
            st.session_state.df = None

    # --- CARGA ---
    def run_uploader(self):
        files = self.upload_view.render()
        if files:
            self.upload_view.show_progress()
            df = self.data_service.load_data(*files)
            if df is not None:
                st.session_state.df = df
                st.rerun()
        
        if st.session_state.df is not None:
            self.upload_view.render_data_preview(st.session_state.df)

    # --- DASHBOARD (ACTUALIZADO) ---
    def run_dashboard(self):
        df = st.session_state.df
        if df is None:
            st.warning("⚠️ Carga datos primero.")
            return
        
        # 1. Capturamos DataFrame filtrado Y nivel de agrupación
        df_filtered, agg_level = self.dashboard_view.render_sidebar_filters(df)
        
        if df_filtered.empty:
            st.warning("⚠️ No hay datos con los filtros actuales.")
            return

        # 2. Renderizamos usando esos parámetros
        self.dashboard_view.render_kpis(df_filtered)
        # Pasamos agg_level para que el gráfico de evolución cambie (Diario/Mensual...)
        self.dashboard_view.render_all_charts(df_filtered, agg_level)

    # --- PREDICCIÓN ---
    def run_prediction(self):
        df = st.session_state.df
        if df is None:
            st.warning("⚠️ Carga datos primero.")
            return

        self.pred_view.render_header()
        
        try:
            agg_hist = self.ml_service.prepare_historical_data(df)
            mae, rmse, r2 = self.ml_service.evaluate_model(agg_hist)
            self.pred_view.render_metrics_test(mae, rmse, r2)
            
            future_2025 = self.ml_service.predict_2025(agg_hist)
            
            if future_2025 is not None:
                self.pred_view.render_forecast_chart(agg_hist, future_2025)
                
                save, name, desc = self.pred_view.render_save_form()
                if save:
                    user = st.session_state.get("username", "Admin")
                    now = datetime.now().isoformat()
                    params = json.dumps({'modelo': 'brutos_v1', 'fecha': now})
                    self.db_service.save_prediction_report(
                        user, now, name, "Predicción", "pred_2025.csv", params, desc
                    )
                    toast("Guardado exitosamente", "💾")
            else:
                st.error("Error generando predicción.")
                
        except Exception as e:
            st.error(f"Error en predicción: {e}")

    # --- ESTRATEGIA ---
    def run_strategy(self):
        df = st.session_state.df
        if df is None:
            st.warning("⚠️ Carga datos para usar Estrategia.")
            return

        tab1, tab2, tab3 = self.strategy_view.render_tabs()

        with tab1: # Markov
            matrix = self.ml_service.get_transition_matrix(df)
            if not matrix.empty:
                self.strategy_view.render_markov_scenario(matrix)
            else:
                st.warning("Faltan datos de transición.")

        with tab2: # Riesgo
            risk = self.ml_service.calculate_risk_var(df)
            if risk:
                self.strategy_view.render_risk_scenario(risk)
            else:
                st.warning("Faltan datos de riesgo.")

        with tab3: # Segmentación
            segs = self.ml_service.get_demographic_segments(df)
            self.strategy_view.render_segmentation_scenario(segs)