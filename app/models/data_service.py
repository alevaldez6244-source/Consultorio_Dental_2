# Archivo: app/models/data_service.py
import pandas as pd
import streamlit as st

class DataService:
    """Encargado de la Extracción, Transformación y Carga (ETL) de datos."""

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_data(areas_io, pac_io, trat_io, aten_io):
        """
        Fusiona los 4 CSVs en un solo DataFrame maestro.
        """
        try:
            # 1. Carga
            areas = pd.read_csv(areas_io, header=None, names=['area_id','area_name'])
            pac   = pd.read_csv(pac_io, header=None, 
                                names=['patient_id','patient_name','age','gender','registration_date'])
            trat  = pd.read_csv(trat_io, header=None, 
                                names=['treatment_id','treatment_name','area_id','price'])
            aten  = pd.read_csv(aten_io, header=None, 
                                names=['attention_id','patient_id','treatment_id',
                                       'attention_date','payment_type','status','extra']
                               ).drop(columns=['extra'], errors='ignore')

            # 2. Transformación (Merge)
            df = (aten.merge(pac,  on='patient_id', how='left')
                       .merge(trat, on='treatment_id', how='left')
                       .merge(areas,on='area_id',     how='left'))

            # 3. Limpieza de Fechas
            df['attention_date'] = pd.to_datetime(df['attention_date'])
            df['year_month']     = df['attention_date'].dt.to_period('M').dt.to_timestamp()
            df['year']           = df['attention_date'].dt.year
            
            return df
        except Exception as e:
            st.error(f"Error en ETL: {e}")
            return None