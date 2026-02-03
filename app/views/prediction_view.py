import streamlit as st
import pandas as pd

class PredictionView:
    def render_header(self):
        st.subheader("🔮 Predicción de Ingresos (IA)")

    def render_metrics_test(self, mae, rmse, r2):
        """Muestra la precisión del modelo."""
        st.markdown("### Métricas de Evaluación del Modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (Error Medio Absoluto)", f"{mae:,.0f}")
        c2.metric("RMSE (Error Cuadrático Medio)", f"{rmse:,.0f}")
        c3.metric("R² (Coeficiente de Determinación)", f"{r2:.3f}")

    def render_forecast_chart(self, history_df, future_df):
        """Gráfico combinado: Historia + Predicción."""
        st.markdown("### Proyección Histórica y Futura (2025)")
        
        # Unir para visualizar
        hist_chart = history_df.set_index('fecha_mes')['ingreso_total']
        fut_chart = future_df.set_index('fecha_mes')['predicted_income']
        
        # Crear un DF conjunto para el gráfico
        chart_data = pd.DataFrame({
            "Histórico": hist_chart,
            "Pronóstico 2025": fut_chart
        })
        
        st.line_chart(chart_data)

    def render_save_form(self):
        """Formulario para guardar el reporte en BD (CRUD)."""
        st.markdown("---")
        with st.expander("💾 Guardar Reporte en Base de Datos"):
            with st.form("save_report"):
                nombre = st.text_input("Nombre del Reporte", "Reporte Mensual")
                desc = st.text_area("Notas adicionales")
                submitted = st.form_submit_button("Guardar")
                return submitted, nombre, desc