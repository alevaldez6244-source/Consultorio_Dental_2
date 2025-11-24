import pandas as pd
import csv
from pathlib import Path

# --- Rutas ---
src = Path(r"C:\Users\Perydox\Desktop\Consultorio Dental\Proyecto_Grado - copia\data\Datos_final\pacientes_final.csv")
dst = src.with_name("pacientes_final_privacidad_clientes.csv")  # salida (copia)

# --- Configuración (ajusta si lo sabes) ---
# Si conoces el índice 0-based de la columna de nombre(s), ponlo aquí. Si no, deja None.
NAME_COL_IDX = None            # p.ej. 1
# Si conoces el/los encabezados de nombre(s), ponlos aquí (insensible a mayúsculas).
NAME_HEADERS = {"paciente","pacientes","nombre","nombres","apellido","apellidos","nombre_paciente","cliente"}

MASK_VALUE = "######"          # lo que se escribirá en la columna de nombres

# --- Detectar delimitador (,, ;, tab, |) ---
def detect_delimiter(path: Path, default=","):
    sample = path.read_text(encoding="utf-8", errors="ignore")[:20000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"]).delimiter
    except Exception:
        return default

sep = detect_delimiter(src)

# --- Leer CSV con fallback de encoding ---
def read_any(path: Path, sep: str):
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=sep, encoding="latin-1", header=0)
    # Si viene todo en una sola columna o sin headers útiles, reintenta sin header
    if df.shape[1] == 1 or all(str(c).isdigit() for c in df.columns):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=sep, encoding="latin-1", header=None)
    return df

df = read_any(src, sep)

# Si quedó en una sola columna, expandimos por el separador
if df.shape[1] == 1:
    df = df.iloc[:, 0].astype(str).str.split(sep, expand=True)

# --- Identificar columna(s) a enmascarar ---
cols = list(df.columns)
cols_lower = [str(c).strip().lower() for c in cols]

to_mask = set()

# 1) Por índice conocido
if NAME_COL_IDX is not None and 0 <= NAME_COL_IDX < len(cols):
    to_mask.add(cols[NAME_COL_IDX])

# 2) Por encabezado (si existen)
if not to_mask:
    for c, cl in zip(cols, cols_lower):
        if (cl in NAME_HEADERS) or ("pacient" in cl) or ("nombre" in cl) or ("apellido" in cl):
            to_mask.add(c)

# 3) Si aún nada y sospechamos formato típico, asumir col 1 como nombre
if not to_mask and len(cols) >= 2:
    to_mask.add(cols[1])

# --- Enmascarar (sin quitar columnas) ---
for c in to_mask:
    df[c] = MASK_VALUE

# --- Guardar CSV (misma carpeta), con BOM para acentos correctos en Excel ---
df.to_csv(dst, index=False, sep=sep, encoding="utf-8-sig")

print(f"Separador detectado: {sep!r}")
print("Columnas enmascaradas:", list(to_mask))
print("Original intacto:", src)
print("Copia con privacidad:", dst)
