import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DashboardView:
    def __init__(self):
        # Configuración estética global
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['figure.figsize'] = (10, 5)

    def render_sidebar_filters(self, df):
        """
        Barra lateral con filtros inteligentes.
        Retorna: (df_filtrado, nivel_agrupacion)
        """
        st.sidebar.header("🔍 Filtros de Análisis")
        
        # 1. Rango de Fechas
        min_date = df['attention_date'].min()
        max_date = df['attention_date'].max()
        fechas = st.sidebar.date_input("📅 Período", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        st.sidebar.markdown("---")
        
        # 2. Filtros Multiselect
        # Áreas
        all_areas = sorted(df['area_name'].dropna().unique())
        areas = st.sidebar.multiselect("🦷 Especialidades", all_areas, default=all_areas)
        
        # Género (Solo mostramos M/F en el selector para limpieza)
        if 'gender' in df.columns:
            valid_genders = [g for g in df['gender'].unique() if str(g).strip().upper() in ['M', 'F', 'MASCULINO', 'FEMENINO']]
            generos = st.sidebar.multiselect("🚻 Género", valid_genders, default=valid_genders)
        else:
            generos = []

        # Estado Pago
        estados_opts = sorted(df['status'].dropna().unique()) if 'status' in df.columns else []
        estados = st.sidebar.multiselect("💰 Estado Pago", estados_opts, default=estados_opts)
        
        st.sidebar.markdown("---")
        
        # 3. Agrupación Temporal
        agg_level = st.sidebar.selectbox(
            "📉 Ver Evolución por:", 
            ['Mensual', 'Diario', 'Semanal', 'Anual'], 
            index=0
        )

        # --- MOTOR DE FILTRADO ---
        if len(fechas) != 2:
            start, end = min_date, max_date
        else:
            start, end = fechas

        # Máscara base
        mask = (
            (df['attention_date'].dt.date >= start) & 
            (df['attention_date'].dt.date <= end) &
            (df['area_name'].isin(areas))
        )
        
        # Filtros opcionales
        if 'gender' in df.columns and generos:
            mask &= (df['gender'].isin(generos))
        if 'status' in df.columns and estados:
            mask &= (df['status'].isin(estados))
            
        return df[mask], agg_level

    def render_kpis(self, df):
        """KPIs estilo tarjeta."""
        st.markdown("### 📊 Tablero de Control")
        c1, c2, c3, c4 = st.columns(4)
        
        total_ingresos = df['price'].sum()
        total_atenciones = len(df)
        ticket_promedio = df['price'].mean() if not df.empty else 0
        pacientes = df['patient_id'].nunique()
        
        c1.metric("Ingresos Totales", f"Bs. {total_ingresos:,.0f}")
        c2.metric("Citas Atendidas", f"{total_atenciones:,.0f}")
        c3.metric("Ticket Promedio", f"Bs. {ticket_promedio:,.0f}")
        c4.metric("Pacientes Activos", f"{pacientes}")
        st.markdown("---")

    def render_all_charts(self, df, agg_level):
        """
        Dashboard organizado simétricamente (2 columnas).
        """
        if df.empty:
            st.warning("⚠️ No hay datos para mostrar con estos filtros.")
            return

        # Pestañas temáticas
        tab1, tab2, tab3 = st.tabs(["📈 Negocio y Tendencias", "🦷 Operativo y Calidad", "👥 Perfil del Paciente"])

        # --- TAB 1: NEGOCIO ---
        with tab1:
            col1, col2 = st.columns(2)
            
            # 1. Evolución Comparativa
            with col1:
                st.subheader(f"Evolución Temporal ({agg_level})")
                
                # Resample dinámico
                rule = {'Diario':'D', 'Semanal':'W', 'Mensual':'M', 'Anual':'Y'}[agg_level]
                evo = df.set_index('attention_date').resample(rule).size()
                
                fig1, ax1 = plt.subplots()
                evo.plot(kind='line', marker='o', ax=ax1, color='#1f77b4', linewidth=2)
                ax1.set_ylabel("Cantidad de Citas")
                ax1.set_xlabel("")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)

            # 2. Mapa de Calor Estacional (Con Leyenda Explícita)
            with col2:
                st.subheader("Patrones de Ingresos (Estacionalidad)")
                try:
                    pivot = df.pivot_table(index=df['attention_date'].dt.year, 
                                          columns=df['attention_date'].dt.month, 
                                          values='price', aggfunc='sum')
                    fig2, ax2 = plt.subplots()
                    # Cbar=True habilita la leyenda de colores
                    sns.heatmap(pivot, cmap="Greens", cbar=True, annot=False, ax=ax2)
                    ax2.set_ylabel("Año")
                    ax2.set_xlabel("Mes")
                    st.pyplot(fig2)
                    st.caption("ℹ️ **Guía:** Los colores más oscuros (verde intenso) indican meses con mayores ingresos económicos.")
                except:
                    st.info("Datos insuficientes para generar el mapa anual.")

            st.divider()
            
            col3, col4 = st.columns(2)
            
            # 3. Ingresos por Área (Ranking Financiero)
            with col3:
                st.subheader("Ranking de Ingresos por Especialidad")
                rev = df.groupby('area_name')['price'].sum().sort_values()
                fig3, ax3 = plt.subplots()
                rev.plot(kind='barh', color='#2ca02c', ax=ax3)
                ax3.set_xlabel("Bolivianos (Bs.)")
                # Etiquetas de valor en las barras
                for i, v in enumerate(rev):
                    ax3.text(v, i, f" {v:,.0f}", va='center', fontsize=9, fontweight='bold')
                st.pyplot(fig3)
            
            # 4. Pareto (80/20)
            with col4:
                st.subheader("Concentración de Ingresos (Pareto)")
                pat_rev = df.groupby('patient_id')['price'].sum().sort_values(ascending=False)
                if not pat_rev.empty:
                    cumsum = pat_rev.cumsum() / pat_rev.sum() * 100
                    x = np.linspace(0, 100, len(pat_rev))
                    
                    fig4, ax4 = plt.subplots()
                    ax4.plot(x, cumsum, color='purple', linewidth=2)
                    ax4.axhline(80, color='red', linestyle='--', alpha=0.5)
                    ax4.text(5, 82, "Corte 80% Ingresos", color='red', fontsize=9)
                    ax4.set_xlabel("% de Pacientes")
                    ax4.set_ylabel("% Ingresos Acumulados")
                    ax4.fill_between(x, cumsum, alpha=0.1, color='purple')
                    st.pyplot(fig4)
                else:
                    st.info("Sin datos para Pareto.")

        # --- TAB 2: OPERATIVO ---
        with tab2:
            col5, col6 = st.columns(2)
            
            # 5. Atenciones por Área (NUEVO)
            with col5:
                st.subheader("Volumen de Atenciones por Área")
                vol_area = df['area_name'].value_counts().sort_values()
                fig5, ax5 = plt.subplots()
                # Color naranja para diferenciar de ingresos
                vol_area.plot(kind='barh', color='#ff7f0e', ax=ax5)
                ax5.set_xlabel("Cantidad de Citas")
                for i, v in enumerate(vol_area):
                    ax5.text(v, i, f" {v}", va='center', fontsize=9)
                st.pyplot(fig5)
            
            # 6. Top Tratamientos
            with col6:
                st.subheader("Top 10 Tratamientos Más Solicitados")
                top10 = df['treatment_name'].value_counts().nlargest(10)
                fig6, ax6 = plt.subplots()
                sns.barplot(x=top10.values, y=top10.index, palette="viridis", ax=ax6)
                ax6.set_xlabel("Frecuencia")
                st.pyplot(fig6)
            
            st.divider()
            
            # 7. Heatmap Área vs Tratamiento (NUEVO)
            st.subheader("Mapa de Densidad: Tratamientos por Especialidad")
            try:
                # Pivot dinámico
                pivot_at = df.pivot_table(index='treatment_name', columns='area_name', values='attention_id', aggfunc='count', fill_value=0)
                
                # Filtramos para mostrar solo los tratamientos relevantes (Top 15 por volumen total)
                top_trats = pivot_at.sum(axis=1).nlargest(15).index
                pivot_at = pivot_at.loc[top_trats]
                
                fig7, ax7 = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot_at, cmap="Blues", annot=True, fmt=".0f", cbar=False, ax=ax7)
                ax7.set_ylabel("")
                ax7.set_xlabel("")
                st.pyplot(fig7)
                st.caption("ℹ️ **Lectura:** Este mapa muestra qué tratamientos específicos se realizan más dentro de cada área. Los números indican cantidad de citas.")
            except:
                st.info("Datos insuficientes para generar el mapa de tratamientos.")

        # --- TAB 3: PACIENTES ---
        with tab3:
            col8, col9 = st.columns(2)
            
            # 8. Género (LIMPIEZA DE DATOS)
            with col8:
                st.subheader("Distribución por Género")
                if 'gender' in df.columns:
                    # Filtramos estrictamente M y F
                    mask_gender = df['gender'].astype(str).str.upper().isin(['M', 'F', 'MASCULINO', 'FEMENINO'])
                    df_gen = df[mask_gender]
                    
                    if not df_gen.empty:
                        counts = df_gen['gender'].value_counts()
                        fig8, ax8 = plt.subplots()
                        counts.plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90, ax=ax8)
                        ax8.set_ylabel("")
                        st.pyplot(fig8)
                    else:
                        st.warning("No hay datos de género 'Masculino' o 'Femenino' disponibles.")
                else:
                    st.info("Columna de género no encontrada.")
            
            # 9. Edad
            with col9:
                st.subheader("Distribución de Edad de Pacientes")
                fig9, ax9 = plt.subplots()
                # Eliminamos nulos o ceros para el histograma
                ages = df[df['age'] > 0]['age']
                sns.histplot(ages, bins=15, kde=True, color='orange', ax=ax9)
                ax9.set_xlabel("Edad (Años)")
                st.pyplot(fig9)