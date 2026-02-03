import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MLService:
    def __init__(self):
        # ---------------------------------------------------------
        # 1. CARGA ROBUSTA DEL MODELO (Usando Pathlib)
        # ---------------------------------------------------------
        # Busca la raíz del proyecto dinámicamente, sin importar desde dónde corras el app
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]  # Sube de app/models/ a la raíz
        
        # Ruta exacta a tu modelo "brutos"
        self.model_path = project_root / 'models' / 'models' / 'modelo_final_2025_brutos.joblib'
        
        self.model = None
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                print(f"✅ Modelo cargado: {self.model_path.name}")
            except Exception as e:
                print(f"❌ Error cargando modelo: {e}")
        else:
            print(f"⚠️ AVISO: No se encontró modelo en {self.model_path}")

    # =========================================================
    # PARTE A: PREDICCIÓN (Lógica Original Restaurada)
    # =========================================================
    
    def prepare_historical_data(self, df):
        """Replica EXACTAMENTE tu ingeniería de características original."""
        if 'attention_date' not in df.columns:
            # Fallback por si la columna se llama 'fecha'
            col_fecha = 'fecha' if 'fecha' in df.columns else None
            if col_fecha:
                df['attention_date'] = pd.to_datetime(df[col_fecha])
            else:
                raise ValueError("No se encontró columna de fecha válida.")

        # Agrupación Mensual
        agg_pred = (
            df.groupby(df["attention_date"].dt.to_period("M"))
            .agg(ingreso_total      = ('price','sum'),
                 total_atenciones   = ('attention_id','count'),
                 pacientes_unicos   = ('patient_id', pd.Series.nunique),
                 pct_pagado         = ('status', lambda s: np.mean(s=='Pagado')))
            .reset_index()
        )
        agg_pred['fecha_mes'] = pd.to_datetime(agg_pred['attention_date'].dt.to_timestamp())
        agg_pred.sort_values('fecha_mes', inplace=True, ignore_index=True)

        # Variables Cíclicas (Tu código original)
        agg_pred['sin_mes']  = np.sin(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
        agg_pred['cos_mes']  = np.cos(2*np.pi*(agg_pred['fecha_mes'].dt.month-1)/12)
        agg_pred['anio_cent'] = agg_pred['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()
        
        # Lags y Media Móvil (Vital para el modelo)
        agg_pred['lag1'] = agg_pred['ingreso_total'].shift(1).bfill()
        agg_pred['lag2'] = agg_pred['ingreso_total'].shift(2).bfill()
        agg_pred['ma3']  = agg_pred['ingreso_total'].rolling(3).mean().shift(1).bfill()

        return agg_pred

    def evaluate_model(self, agg_pred):
        """Calcula métricas reales usando tu split 80/20."""
        if self.model is None:
            return 0, 0, 0

        VARIABLES = ['total_atenciones','pacientes_unicos','pct_pagado',
                     'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3']

        # Split idéntico al entrenamiento
        _, test_idx = train_test_split(agg_pred.index, test_size=0.2, random_state=42)
        
        y_test = agg_pred.loc[test_idx, 'ingreso_total']
        X_test = agg_pred.loc[test_idx, VARIABLES]
        
        try:
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            return mae, rmse, r2
        except Exception as e:
            print(f"Error evaluando: {e}")
            return 0, 0, 0

    def predict_2025(self, agg_pred):
        """Genera el forecast 2025 paso a paso."""
        if self.model is None:
            return None

        VARIABLES = ['total_atenciones','pacientes_unicos','pct_pagado',
                     'sin_mes','cos_mes','anio_cent','lag1','lag2','ma3']

        # 1. Crear fechas futuras
        future = pd.DataFrame({'fecha_mes': pd.date_range('2025-01-01','2025-12-01',freq='MS')})
        
        # 2. Ingeniería de características futura
        future['sin_mes']  = np.sin(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
        future['cos_mes']  = np.cos(2*np.pi*(future['fecha_mes'].dt.month-1)/12)
        future['anio_cent']= future['fecha_mes'].dt.year - agg_pred['fecha_mes'].dt.year.min()

        # 3. Lags iniciales (último dato conocido)
        last_row = agg_pred.iloc[-1]
        for lag in ['lag1','lag2','ma3']:
            future[lag] = last_row[lag]

        # 4. Promedios para variables operativas
        grp = agg_pred.groupby(agg_pred['fecha_mes'].dt.month)
        future['total_atenciones'] = future['fecha_mes'].dt.month.map(grp['total_atenciones'].mean())
        future['pacientes_unicos'] = future['fecha_mes'].dt.month.map(grp['pacientes_unicos'].mean())
        future['pct_pagado']       = future['fecha_mes'].dt.month.map(grp['pct_pagado'].mean())

        # 5. Predicción
        try:
            future['predicted_income'] = self.model.predict(future[VARIABLES])
        except Exception as e:
            print(f"Error predicción 2025: {e}")
            return None

        return future

    # =========================================================
    # PARTE B: ESTRATEGIA (Nuevos Escenarios)
    # =========================================================
    
    def get_transition_matrix(self, df):
        """Escenario 1: Cadenas de Markov (Flujo de Pacientes)"""
        df_sorted = df.sort_values(by=['patient_id', 'attention_date'])
        df_sorted['prev_area'] = df_sorted.groupby('patient_id')['area_name'].shift(1)
        transitions = df_sorted.dropna(subset=['prev_area'])
        
        if transitions.empty: return pd.DataFrame()
        return pd.crosstab(transitions['prev_area'], transitions['area_name'], normalize='index')

    def calculate_risk_var(self, df, simulations=10000):
        """Escenario 2: Value at Risk (Riesgo Financiero)"""
        df['ym'] = df['attention_date'].dt.to_period('M')
        monthly_income = df.groupby('ym')['price'].sum()
        
        if len(monthly_income) < 2: return None
        
        mu = monthly_income.mean()
        sigma = monthly_income.std()
        
        # Simulación Monte Carlo
        simulated_months = np.random.normal(mu, sigma, simulations)
        var_95 = np.percentile(simulated_months, 5) # Peor 5% de los casos
        
        return {
            'mean_income': mu,
            'std_dev': sigma,
            'var_95': var_95,
            'simulated_data': simulated_months
        }

    def get_demographic_segments(self, df):
        """Escenario 3: Segmentación Pareto (ROI por edad)"""
        # Limpieza de edad a prueba de errores
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