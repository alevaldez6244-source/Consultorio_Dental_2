import pandas as pd
from pathlib import Path

# Ruta del CSV ORIGINAL (no se modifica)
src = Path(r"C:\Users\Perydox\Desktop\Consultorio Dental\Proyecto_Grado - copia\data\Datos_final\pacientes_final.csv")

# Nombre del archivo de salida (copia limpia) en la MISMA carpeta
dst = src.with_name("pacientes_final_privacidad_clientes.xlsx")

# (Opcional) especifica columnas exactas si las conoces; si no, el script intentará detectarlas
COLUMNS_TO_DROP_EXACT = []  # ej.: ["Paciente", "Pacientes", "Nombre", "Apellidos"]

# Cargar CSV (detección automática de separador; fallback de encoding)
try:
    df = pd.read_csv(src, sep=None, engine="python", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(src, sep=None, engine="python", encoding="latin-1")

cols = list(df.columns)

# Detección automática de columnas sensibles (nombre del paciente)
patterns = ("pacient", "cliente", "nombre", "apellido")  # ajusta si deseas ser más/menos agresivo
auto_drop = [c for c in cols if any(p in c.lower() for p in patterns)]

# Si definiste COLUMNS_TO_DROP_EXACT, se priorizan; si no, usa la detección automática
cols_to_drop = COLUMNS_TO_DROP_EXACT if COLUMNS_TO_DROP_EXACT else auto_drop

# Crear copia sin columnas sensibles (NO modifica el original)
df_out = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

# Guardar como Excel en la misma ruta
df_out.to_excel(dst, index=False)

print("Columnas eliminadas:", [c for c in cols_to_drop if c in cols])
print("Original intacto en:", src)
print("Copia limpia guardada en:", dst)
