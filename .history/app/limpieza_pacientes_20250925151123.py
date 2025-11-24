import pandas as pd
import csv
from pathlib import Path

# --- Rutas ---
src = Path(r"C:\Users\Perydox\Desktop\Consultorio Dental\Proyecto_Grado - copia\data\Datos_final\pacientes_final.csv")
dst = src.with_name("pacientes_final_privacidad_clientes.csv")  # salida (copia)

# --- Detectar delimitador (,, ;, tab, |) ---
def detect_delimiter(path: Path, default=","):
    sample = path.read_text(encoding="utf-8", errors="ignore")[:20000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"]).delimiter
    except Exception:
        return default

sep = detect_delimiter(src)

# --- Intentar leer (con y sin encabezado) ---
def read_any(path: Path, sep: str):
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=sep, encoding="latin-1", header=0)
    # Si solo hay 1 columna o los encabezados no son útiles, leemos sin header
    if df.shape[1] == 1 or set(map(str.lower, df.columns)) == set(range(df.shape[1])):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=sep, encoding="latin-1", header=None)
    return df

df = read_any(src, sep)

# Si quedó en una sola columna, la expandimos manualmente
if df.shape[1] == 1:
    df = df.iloc[:,0].astype(str).str.split(sep, expand=True)

# ---- Identificar la columna de NOMBRE ----
# Muchos archivos con ese formato suelen ser: [id, nombre, edad, sexo, ...]
# Por defecto asumimos que la columna de NOMBRE es la 1.
NAME_COL_IDX = 1

# Si hay encabezados y alguno suena a nombre/paciente, usamos ese
name_like = {"paciente","pacientes","nombre","nombres","nombre_paciente","id_paciente"}
header_lower = list(map(lambda x: str(x).strip().lower(), df.columns))
cands = [i for i, c in enumerate(header_lower) if (c in name_like) or ("pacient" in c)]
if cands:
    NAME_COL_IDX = cands[0]

# Quitar la columna de nombre (si existe ese índice)
cols = list(df.columns)
if 0 <= NAME_COL_IDX < len(cols):
    df = df.drop(columns=cols[NAME_COL_IDX], errors="ignore")

# --- Guardar CSV (misma carpeta), con BOM para que Excel muestre bien acentos ---
df.to_csv(dst, index=False, sep=sep, encoding="utf-8-sig")

print(f"Separador detectado: {sep!r}")
print("Archivo original intacto:", src)
print("Copia limpia guardada en:", dst)
