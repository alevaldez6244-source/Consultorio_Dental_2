import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MLService:
    def __init__(self):
        # ---------------------------------------------------------
        # CARGA ROBUSTA DEL MODELO (Pathlib)
        # ---------------------------------------------------------
        # Busca la raíz del proyecto dinámicamente
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]  # Sube 2 niveles hasta la raíz
        
        # Ruta construida: root/models/models/modelo_final_2025_brutos.joblib
        self.model_path = project_root / 'models' / 'models' / 'modelo_final_2025_brutos.joblib'
        
        self.model = None
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                print(f"✅ Modelo cargado desde: {self.model_path.name}")
            except Exception as e:
                print(f"❌ Error cargando modelo: {e}")
        else:
            print(f"⚠️ AVISO: Modelo no encontrado en {self.model_path}")

    # =========================================================
    # PARTE A: PREDICCIÓN (IA)
    # =========================================================
    
    def prepare_historical_data(self, df):
        """Prepara los datos históricos (Agrupación + Features)."""
        if 'attention_date' not in df.columns:
            if 'fecha' in df.columns:
                df['attention_date'] = pd.to_datetime(df['fecha'])
            else:
                raise ValueError("El dataset no tiene columna de fecha ('attention_date' o 'fecha')")

        # Agrupación Mensual
        agg = (
            df.groupby(df["attention_date"].dt.to_period("M"))
            .agg(ingreso_total=('price','sum'),
                 total_atenciones=('attention_id','count'),
                 pacientes_unicos=('patient_id', pd.Series.nunique),
                 pct_pagado=('status', lambda s: np.mean(s=='Pagado')))
            .reset_index()
        )
        agg['fecha_mes'] = agg['attention_date'].dt.to_timestamp()
        agg.sort_values('fecha_mes', inplace=True, ignore_index=True)

        # Feature Engineering (Cíclicas + Lags)
        agg['sin_mes']  = np.sin(2*np.pi*(agg['fecha_mes'].dt.month-1)/12)
        agg['cos_mes']  = np.cos(2*np.pi*(agg['fecha_mes'].dt.month-1)/12)
        agg['anio_cent'] = agg['fecha_mes'].dt.year - agg['fecha_mes'].dt.year.min()
        
        agg['lag1'] = agg['ingreso_total'].shift(1).bfill()
        agg['lag2'] = agg['ingreso_total'].shift(2).bfill()
        agg['ma3']  = agg['ingreso_total'].rolling(3).mean().shift(1).bfill()

        return agg

    def evaluate_model(self, agg_pred):
        """
        Retorna Métricas + Datos de Validación (Test Set) para graficar.
        """
        if self.model is None:
            return 0, 0, 0, None

        VARIABLES = ['total_atenciones','pacientes_unicos','pct_pagado',
                     'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3']

        # Split 80/20 (Mismo random_state que el entrenamiento para consistencia)
        _, test_idx = train_test_split(agg_pred.index, test_size=0.2, random_state=42)
        
        # Creamos un DataFrame solo con el Test Set para validación visual
        val_df = agg_pred.loc[test_idx].copy().sort_values('fecha_mes')
        
        X_test = val_df[VARIABLES]
        y_test = val_df['ingreso_total']
        
        try:
            # Predecimos sobre los datos de prueba
            y_pred = self.model.predict(X_test)
            
            # Guardamos la predicción en el DF para graficarla luego
            val_df['predicted_income'] = y_pred
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            return mae, rmse, r2, val_df
            
        except Exception as e:
            print(f"Error evaluando modelo: {e}")
            return 0, 0, 0, None

    def predict_2025(self, agg_pred):
        """Genera la proyección futura 2025."""
        if self.model is None: return None

        VARIABLES = ['total_atenciones','pacientes_unicos','pct_pagado',
                     'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3']

        # 1. Fechas Futuras
        future = pd.DataFrame({'fecha_mes': pd.date_range('2025-01-01','2025-12-01',freq='MS')})
        
        # 2. Features Temporales
        future['sin_mes']  = np.sin(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
        future['cos_mes']  = np.cos(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
        future['anio_cent']= future['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()

        # 3. Lags (Inicializados con el último dato real)
        last_row = agg_pred.iloc[-1]
        for lag in ['lag1','lag2','ma3']:
            future[lag] = last_row[lag]

        # 4. Promedios mensuales para variables operativas
        grp = agg_pred.groupby(agg_pred['fecha_mes'].dt.month)
        future['total_atenciones'] = future['fecha_mes'].dt.month.map(grp['total_atenciones'].mean())
        future['pacientes_unicos'] = future['fecha_mes'].dt.month.map(grp['pacientes_unicos'].mean())
        future['pct_pagado']       = future['fecha_mes'].dt.month.map(grp['pct_pagado'].mean())

        try:
            future['predicted_income'] = self.model.predict(future[VARIABLES])
            return future
        except Exception as e:
            print(f"Error en predicción 2025: {e}")
            return None

    # =========================================================
    # PARTE B: ESTRATEGIA (Módulos de Negocio)
    # =========================================================
    
    def get_transition_matrix(self, df):
        """Markov: Probabilidad de transición entre áreas."""
        df_sorted = df.sort_values(by=['patient_id', 'attention_date'])
        df_sorted['prev_area'] = df_sorted.groupby('patient_id')['area_name'].shift(1)
        transitions = df_sorted.dropna(subset=['prev_area'])
        
        if transitions.empty: return pd.DataFrame()
        return pd.crosstab(transitions['prev_area'], transitions['area_name'], normalize='index')

    def calculate_risk_var(self, df, simulations=10000):
        """VaR: Riesgo Financiero."""
        df['ym'] = df['attention_date'].dt.to_period('M')
        monthly_income = df.groupby('ym')['price'].sum()
        
        if len(monthly_income) < 2: return None
        
        mu = monthly_income.mean()
        sigma = monthly_income.std()
        
        simulated_months = np.random.normal(mu, sigma, simulations)
        var_95 = np.percentile(simulated_months, 5)
        
        return {
            'mean_income': mu,
            'std_dev': sigma,
            'var_95': var_95,
            'simulated_data': simulated_months
        }

    def get_demographic_segments(self, df):
        """Pareto: ROI por edad."""
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
        
        bins = [0, 18, 35, 60, 100]
        labels = ['Niños (0-18)', 'Jóvenes (19-35)', 'Adultos (36-60)', 'Senior (60+)']
        
        df_seg = df.copy()
        df_seg['age_group'] = pd.cut(df_seg['age'], bins=bins, labels=labels)
        
        return df_seg.groupby('age_group', observed=False).agg(
            total_revenue=('price', 'sum'),
            patient_count=('patient_id', 'nunique'),
            avg_ticket=('price', 'mean')
        ).reset_index()