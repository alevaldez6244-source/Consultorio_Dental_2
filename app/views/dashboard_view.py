# Archivo: app/views/dashboard_view.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DashboardView:
    def __init__(self):
        # Estilo limpio
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['figure.figsize'] = (10, 5)

    def render_sidebar_filters(self, df):
        """
        Barra lateral con todos los filtros solicitados.
        Retorna: (df_filtrado, nivel_agrupacion)
        """
        st.sidebar.header("🔍 Filtros de Análisis")
        
        # 1. Rango de Fechas
        min_date = df['attention_date'].min()
        max_date = df['attention_date'].max()
        fechas = st.sidebar.date_input("📅 Rango de Fechas", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        st.sidebar.markdown("---")
        
        # 2. Filtros Multiselect
        areas = st.sidebar.multiselect("🦷 Áreas", df['area_name'].unique(), default=df['area_name'].unique())
        
        # Manejo seguro de columnas opcionales (por si el CSV cambia)
        generos_opts = df['gender'].unique() if 'gender' in df.columns else []
        generos = st.sidebar.multiselect("🚻 Género", generos_opts, default=generos_opts)
        
        pagos_opts = df['payment_type'].unique() if 'payment_type' in df.columns else []
        pagos = st.sidebar.multiselect("💳 Tipo de Pago", pagos_opts, default=pagos_opts)
        
        estados_opts = df['status'].unique() if 'status' in df.columns else []
        estados = st.sidebar.multiselect("💰 Estado Pago", estados_opts, default=estados_opts)
        
        st.sidebar.markdown("---")
        
        # 3. Selector de Agrupación (Para el gráfico de Evolución)
        agg_level = st.sidebar.selectbox(
            "📉 Agrupar Evolución por:", 
            ['Mensual', 'Diario', 'Semanal', 'Anual'], 
            index=0
        )

        # --- Lógica de Filtrado ---
        if len(fechas) != 2:
            start, end = min_date, max_date
        else:
            start, end = fechas

        # Máscara booleana base
        mask = (
            (df['attention_date'].dt.date >= start) & 
            (df['attention_date'].dt.date <= end) &
            (df['area_name'].isin(areas))
        )
        
        # Agregar filtros opcionales si existen las columnas
        if 'gender' in df.columns:
            mask &= (df['gender'].isin(generos))
        if 'payment_type' in df.columns:
            mask &= (df['payment_type'].isin(pagos))
        if 'status' in df.columns:
            mask &= (df['status'].isin(estados))
            
        df_filtered = df[mask]
        
        return df_filtered, agg_level

    def render_kpis(self, df):
        """Tarjetas de resumen."""
        st.markdown("### 📊 KPIs Generales")
        c1, c2, c3, c4 = st.columns(4)
        
        total_ingresos = df['price'].sum()
        total_atenciones = len(df)
        ticket_promedio = df['price'].mean() if not df.empty else 0
        pacientes = df['patient_id'].nunique()
        
        c1.metric("Ingresos", f"Bs. {total_ingresos:,.0f}")
        c2.metric("Atenciones", f"{total_atenciones:,.0f}")
        c3.metric("Ticket Promedio", f"Bs. {ticket_promedio:,.0f}")
        c4.metric("Pacientes Únicos", f"{pacientes}")
        st.markdown("---")

    def render_all_charts(self, df, agg_level):
        """
        Renderiza los gráficos usando el 'agg_level' seleccionado.
        """
        if df.empty:
            st.warning("⚠️ No hay datos para mostrar con estos filtros.")
            return

        # Tabs para organizar
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Tendencias", "💰 Finanzas", "🦷 Operativo", "👥 Pacientes"])

        # --- TAB 1: EVOLUCIÓN (Dinámica según agg_level) ---
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Evolución de Atenciones ({agg_level})")
                
                # Lógica de agrupación dinámica
                if agg_level == 'Diario':
                    resample_code = 'D'
                elif agg_level == 'Semanal':
                    resample_code = 'W'
                elif agg_level == 'Mensual':
                    resample_code = 'M'
                else: # Anual
                    resample_code = 'Y'
                
                # Re-muestreo temporal
                evo_data = df.set_index('attention_date').resample(resample_code).size()
                
                fig1, ax1 = plt.subplots()
                evo_data.plot(kind='line', marker='o', ax=ax1, color='#1f77b4', linewidth=2)
                ax1.set_ylabel("Cantidad de Atenciones")
                ax1.set_xlabel("Tiempo")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)

            with col2:
                st.subheader("Mapa de Calor (Año vs Mes)")
                # Solo tiene sentido si hay suficientes datos mensuales
                try:
                    pivot = df.pivot_table(
                        index=df['attention_date'].dt.year, 
                        columns=df['attention_date'].dt.month, 
                        values='price', aggfunc='sum'
                    )
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(pivot, cmap="Greens", cbar=False, annot=False, ax=ax2)
                    ax2.set_title("Intensidad de Ingresos")
                    st.pyplot(fig2)
                except:
                    st.info("Datos insuficientes para Heatmap anual.")

        # --- TAB 2: FINANZAS ---
        with tab2:
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Ingresos por Área")
                rev = df.groupby('area_name')['price'].sum().sort_values()
                fig3, ax3 = plt.subplots()
                rev.plot(kind='barh', color='#2ca02c', ax=ax3)
                ax3.set_xlabel("Bolivianos (Bs.)")
                st.pyplot(fig3)
            
            with c4:
                st.subheader("Medios de Pago")
                if 'payment_type' in df.columns:
                    fig4, ax4 = plt.subplots()
                    df['payment_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax4, startangle=90)
                    ax4.set_ylabel("")
                    st.pyplot(fig4)

        # --- TAB 3: OPERATIVO ---
        with tab3:
            st.subheader("Top 10 Tratamientos")
            top10 = df['treatment_name'].value_counts().nlargest(10)
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            sns.barplot(x=top10.values, y=top10.index, palette="viridis", ax=ax5)
            st.pyplot(fig5)

        # --- TAB 4: PACIENTES ---
        with tab4:
            c5, c6 = st.columns(2)
            with c5:
                st.subheader("Distribución de Edad")
                fig6, ax6 = plt.subplots()
                sns.histplot(df['age'], bins=15, kde=True, color='orange', ax=ax6)
                st.pyplot(fig6)
            
            with c6:
                st.subheader("Género")
                if 'gender' in df.columns:
                    fig7, ax7 = plt.subplots()
                    df['gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax7)
                    ax7.set_ylabel("")
                    st.pyplot(fig7)