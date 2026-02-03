import streamlit as st
import time

class UploadView:
    def render(self):
        st.subheader("📂 Carga de Datos ETL")
        st.markdown("Sube los archivos CSV exportados del sistema administrativo.")

        # Layout de columnas para los uploaders
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            areas = st.file_uploader("Áreas", type="csv", key="u1")
            pacientes = st.file_uploader("Pacientes", type="csv", key="u2")
            tratamientos = st.file_uploader("Tratamientos", type="csv", key="u3")
            atenciones = st.file_uploader("Atenciones", type="csv", key="u4")

        # Botón de acción (solo aparece si todos los archivos están cargados)
        ready = all([areas, pacientes, tratamientos, atenciones])
        
        # Retornamos los archivos si el usuario da click, sino None
        if ready:
            if st.button("🚀 Iniciar Procesamiento", type="primary"):
                return [areas, pacientes, tratamientos, atenciones]
        
        return None

    def show_progress(self):
        """Muestra la animación de carga prolongada."""
        msg = st.empty()
        progress = st.progress(0)
        
        for i in range(10):
            if   i < 3: msg.info("Procesando archivos…")
            elif i < 6: msg.info("Limpiando datos…")
            elif i < 9: msg.info("Optimizando tablas…")
            else:       msg.info("¡Ya casi está listo!")
            
            time.sleep(1) # Simulación de proceso
            progress.progress((i+1)/10)
            
        msg.empty()
        progress.empty()

    def render_data_preview(self, df):
        """Muestra la confirmación y la tabla de vista previa."""
        st.markdown("---")
        
        # 1. Notificación Visual Persistente
        st.success("✅ ¡Los datos se han cargado y unificado correctamente!")
        
        # 2. Vista Rápida
        st.subheader("Vista Rápida (Primeros 20 registros)")
        st.caption("Esta es una muestra de cómo quedaron tus datos unificados:")
        
        # Mostramos el DataFrame
        st.dataframe(df.head(20), use_container_width=True)
        
        # Métricas rápidas de validación
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas Totales", f"{len(df):,.0f}")
        c2.metric("Columnas", f"{len(df.columns)}")
        c3.metric("Rango Fechas", f"{df['attention_date'].min().date()} / {df['attention_date'].max().date()}")