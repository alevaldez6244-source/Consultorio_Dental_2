import streamlit as st
import pandas as pd
import json
from datetime import datetime

# MVC Imports
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

    # --- DASHBOARD (ACTUALIZADO CON FILTROS) ---
    def run_dashboard(self):
        df = st.session_state.df
        if df is None:
            st.warning("⚠️ Carga datos primero.")
            return
        
        # 1. Capturamos DataFrame filtrado Y nivel de agrupación
        # Esto permite que los selectores de la barra lateral afecten a los gráficos
        df_filtered, agg_level = self.dashboard_view.render_sidebar_filters(df)
        
        if df_filtered.empty:
            st.warning("⚠️ No hay datos con los filtros actuales. Intenta ampliar el rango de fechas.")
            return

        # 2. Renderizamos usando esos parámetros
        self.dashboard_view.render_kpis(df_filtered)
        self.dashboard_view.render_all_charts(df_filtered, agg_level)

    # --- PREDICCIÓN ---
    def run_prediction(self):
        df = st.session_state.df
        if df is None:
            st.warning("⚠️ Carga datos primero.")
            return

        self.pred_view.render_header()
        
        try:
            # 1. Preparar datos históricos
            agg_hist = self.ml_service.prepare_historical_data(df)
            
            # 2. Obtener métricas Y Datos de Validación (Test Set)
            # Esto es clave para el gráfico de "Prueba y Error"
            mae, rmse, r2, val_df = self.ml_service.evaluate_model(agg_hist)
            
            self.pred_view.render_metrics_test(mae, rmse, r2)
            
            # 3. Generar Forecast 2025
            future = self.ml_service.predict_2025(agg_hist)
            
            if future is not None:
                # Pasamos las 3 piezas al gráfico: Historia, Validación y Futuro
                self.pred_view.render_forecast_chart(agg_hist, val_df, future)
                
                # Guardado en BD
                save, name, desc = self.pred_view.render_save_form()
                if save:
                    user = st.session_state.get("username", "Admin")
                    now = datetime.now().isoformat()
                    params = json.dumps({'modelo': 'brutos_v1', 'fecha': now})
                    
                    self.db_service.save_prediction_report(
                        user, now, name, "Predicción", "pred_2025.csv", params, desc
                    )
                    toast("Reporte guardado exitosamente", "💾")
            else:
                st.error("Error al generar la proyección futura.")
                
        except Exception as e:
            st.error(f"Error en módulo de predicción: {e}")

    # --- ESTRATEGIA ---
    def run_strategy(self):
        df = st.session_state.df
        if df is None:
            st.warning("⚠️ Carga datos primero.")
            return

        # Renderizar Tabs
        t1, t2, t3 = self.strategy_view.render_tabs()
        
        with t1: # Markov
            mat = self.ml_service.get_transition_matrix(df)
            if not mat.empty:
                self.strategy_view.render_markov_scenario(mat)
            else:
                st.warning("Faltan datos históricos para calcular transiciones.")
            
        with t2: # Riesgo
            risk = self.ml_service.calculate_risk_var(df)
            if risk:
                self.strategy_view.render_risk_scenario(risk)
            else:
                st.warning("Faltan datos para análisis de riesgo.")
            
        with t3: # Segmentación
            seg = self.ml_service.get_demographic_segments(df)
            self.strategy_view.render_segmentation_scenario(seg)