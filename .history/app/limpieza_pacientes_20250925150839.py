import pandas as pd
import csv
from pathlib import Path

# --- Rutas ---
src = Path(r"C:\Users\Perydox\Desktop\Consultorio Dental\Proyecto_Grado - copia\data\Datos_final\pacientes_final.csv")
dst = src.with_name("pacientes_final_privacidad_clientes.csv")  # salida CSV

# (Opcional) si sabes el nombre exacto, ponlo aquí; si no, déjalo vacío y se detecta
COLUMNS_TO_DROP_EXACT = []  # ej.: ["Pacientes"]

# --- Detectar delimitador (,, ;, tab, |) ---
def detect_delimiter(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(100000)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return ","  # fallback

sep = detect_delimiter(src)

# --- Cargar CSV (con fallback de encoding) ---
try:
    df = pd.read_csv(src, sep=sep, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(src, sep=sep, encoding="latin-1")

# --- Determinar columnas a eliminar ---
cols = list(df.columns)
cols_auto = [c for c in cols if any(p in c.lower() for p in (
    "pacient",      # paciente/pacientes
    "nombre",       # nombre(s)
    "apellido",     # apellido(s)
    "id_paciente",  # id de paciente
))]
cols_to_drop = (COLUMNS_TO_DROP_EXACT or cols_auto)

# --- Generar copia sin columnas sensibles (NO toca el original) ---
df_out = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

# --- Guardar CSV con el mismo delimitador ---
df_out.to_csv(dst, index=False, sep=sep)

print("Columnas eliminadas:", [c for c in cols_to_drop if c in cols])
print("Original intacto:", src)
print("Copia limpia guardada en:", dst)
