import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class StrategyView:
    def render_tabs(self):
        st.subheader("🧠 Laboratorio de Estrategia")
        st.markdown("Simula escenarios futuros basados en la 'física' de tus datos históricos.")
        
        tab1, tab2, tab3 = st.tabs([
            "🔄 Flujo de Pacientes (Markov)", 
            "📉 Riesgo Financiero (VaR)", 
            "🎯 Segmentación (Pareto)"
        ])
        return tab1, tab2, tab3

    # --- ESCENARIO 1: MARKOV ---
    def render_markov_scenario(self, transition_matrix):
        st.info("Predice: Si un paciente llega al Área A, ¿cuál es la probabilidad de que vaya al Área B?")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Configuración")
            areas = transition_matrix.index.tolist()
            # Intenta seleccionar Odontopediatría por defecto (tu caso de uso real)
            idx_default = areas.index("Odontopediatría") if "Odontopediatría" in areas else 0
            
            start_area = st.selectbox("Área Inicial", areas, index=idx_default)
            num_patients = st.number_input("Pacientes Nuevos", min_value=1, value=50)
            
            # Cálculo
            probs = transition_matrix.loc[start_area]
            predicted = probs * num_patients
            predicted = predicted[predicted > 0.5].sort_values(ascending=False)

        with col2:
            st.markdown(f"#### Proyección de Flujo")
            if not predicted.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=predicted.values, y=predicted.index, palette="viridis", ax=ax)
                ax.set_xlabel("Pacientes Derivados (Estimado)")
                
                # Etiquetas de valor
                for i, v in enumerate(predicted.values):
                    ax.text(v + 0.1, i, f"{v:.1f}", va='center', fontweight='bold')
                st.pyplot(fig)
            else:
                st.warning("No hay suficientes datos de transición para esta área.")

    # --- ESCENARIO 2: RIESGO ---
    def render_risk_scenario(self, risk_data):
        st.info("Prueba de Estrés: ¿Qué probabilidad hay de no cubrir los costos fijos el próximo mes?")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            costs = st.number_input("Costos Fijos Mensuales (Bs.)", value=15000, step=500)
            
            simulated = risk_data['simulated_data']
            prob_loss = (simulated < costs).mean() * 100
            
            st.metric("Probabilidad de Déficit", f"{prob_loss:.1f}%")
            
            if prob_loss > 15:
                st.error("RIESGO ALTO 🚨")
            elif prob_loss > 0:
                st.warning("RIESGO MODERADO ⚠️")
            else:
                st.success("SOLVENCIA SÓLIDA ✅")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(simulated, color="skyblue", kde=True, ax=ax)
            ax.axvline(costs, color='red', linestyle='--', linewidth=2, label=f'Costos (Bs. {costs})')
            ax.axvline(risk_data['var_95'], color='green', linestyle=':', linewidth=2, label=f'Suelo 95% (Bs. {risk_data["var_95"]:,.0f})')
            ax.legend()
            ax.set_title("Distribución Probabilística de Ingresos")
            st.pyplot(fig)

    # --- ESCENARIO 3: SEGMENTACIÓN ---
    def render_segmentation_scenario(self, df_seg):
        st.info("ROI por Edad: ¿Qué grupo generaría más dinero si aumentamos la captación?")
        
        # Tabla formateada
        st.dataframe(
            df_seg.style.format({
                'total_revenue': 'Bs. {:,.0f}',
                'avg_ticket': 'Bs. {:,.2f}'
            }), 
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            segment = st.selectbox("Segmento a Potenciar", df_seg['age_group'].unique())
            growth = st.slider(f"Aumento de Pacientes (%)", 0, 50, 10)
            
        with col2:
            row = df_seg[df_seg['age_group'] == segment].iloc[0]
            extra_pax = int(row['patient_count'] * (growth/100))
            extra_rev = extra_pax * row['avg_ticket']
            
            st.metric(
                label="Ingreso Extra Estimado",
                value=f"Bs. {extra_rev:,.0f}",
                delta=f"+{extra_pax} pacientes"
            )