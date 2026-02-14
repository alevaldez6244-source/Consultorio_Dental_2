import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

class PredictionView:
    def render_header(self):
        st.subheader("🔮 Predicción de Ingresos (IA)")
        st.markdown("""
        **Modelo:** Random Forest Regressor  
        **Horizonte:** Proyección 2025 basada en patrones históricos.
        """)

    def render_metrics_test(self, mae, rmse, r2):
        st.markdown("### 🎯 Precisión del Modelo (Fase de Validación)")
        st.info("Estas métricas indican qué tan bien aprendió el modelo usando los datos de prueba (2024).")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Error Medio (MAE)", f"Bs. {mae:,.0f}", help="En promedio, el modelo falla por esta cantidad.")
        c2.metric("Desviación (RMSE)", f"Bs. {rmse:,.0f}", help="Castiga más los errores grandes.")
        
        # Color semántico para R2
        delta_color = "normal" if r2 >= 0.7 else "off"
        c3.metric("Ajuste (R²)", f"{r2*100:.1f}%", delta_color=delta_color, help="Porcentaje de la variabilidad explicada por el modelo (Ideal > 70%).")

    def render_forecast_chart(self, history_df, validation_df, forecast_df):
        """
        Gráfico Maestro: Historia + Validación (Real vs Pred) + Futuro
        """
        st.subheader("📉 Proyección: Historia, Validación y Futuro")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Historia Real (Gris) - Contexto
        # Usamos los datos históricos completos para dar contexto visual
        ax.plot(history_df['fecha_mes'], history_df['ingreso_total'], 
                label='Histórico (2021-2023)', color='gray', alpha=0.4, linewidth=2)
        
        # 2. Validación / Prueba (Puntos Azules vs Rojos) - La "Prueba de Fuego"
        # Esto muestra explícitamente el "Test Set" (aprox 2024)
        if validation_df is not None and not validation_df.empty:
            ax.plot(validation_df['fecha_mes'], validation_df['ingreso_total'], 
                    'o', color='blue', label='Real 2024 (Test)', markersize=6, alpha=0.7)
            ax.plot(validation_df['fecha_mes'], validation_df['predicted_income'], 
                    'x', color='red', label='Predicción Modelo 2024', markersize=8, markeredgewidth=2)
            
            # Dibujar líneas verticales pequeñas conectando Real vs Predicho (Residuales visuales)
            for idx, row in validation_df.iterrows():
                ax.vlines(row['fecha_mes'], row['ingreso_total'], row['predicted_income'], 
                          colors='red', linestyles=':', alpha=0.3)

        # 3. Futuro 2025 (Verde) - La Proyección
        # Conectamos visualmente con el último punto histórico si es posible
        ax.plot(forecast_df['fecha_mes'], forecast_df['predicted_income'], 
                label='Pronóstico 2025', color='#4CAF50', linestyle='--', linewidth=3, marker='o')
        
        # Estética del gráfico
        ax.set_title("Línea de Tiempo Completa de Ingresos")
        ax.set_ylabel("Ingresos Mensuales (Bs.)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Formato de miles en eje Y (Ej: 20,000)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        st.pyplot(fig)
        
        # Tabla de Datos Futuros
        with st.expander("Ver Tabla de Datos (Proyección 2025)"):
            disp = forecast_df[['fecha_mes', 'predicted_income']].copy()
            disp.columns = ['Mes', 'Ingreso Proyectado (Bs.)']
            # Formateamos la fecha para que sea legible
            disp['Mes'] = disp['Mes'].dt.strftime('%B %Y')
            st.dataframe(disp.style.format({'Ingreso Proyectado (Bs.)': 'Bs. {:,.2f}'}), use_container_width=True)

    def render_save_form(self):
        st.markdown("### 💾 Guardar Reporte")
        with st.form("save_pred"):
            name = st.text_input("Nombre del Reporte", value="Proyección Financiera 2025")
            desc = st.text_area("Notas / Observaciones", placeholder="Ej: Escenario base considerando aumento de pacientes...")
            submit = st.form_submit_button("Guardar en Base de Datos")
            return submit, name, desc