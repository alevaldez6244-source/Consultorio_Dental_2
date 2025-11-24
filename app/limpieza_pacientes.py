import pandas as pd
import csv
from pathlib import Path

src = Path(r"C:\Users\Perydox\Desktop\Consultorio Dental\Proyecto_Grado - copia\data\Datos_final\pacientes_final.csv")
dst = src.with_name("pacientes_final_privacidad_clientes.csv")

MASK_VALUE = "######"
NAME_COL_IDX = 1  # la columna de nombre (0-based). Ajusta si es otra.

# Detectar delimitador
def detect_delimiter(path: Path, default=","):
    sample = path.read_text(encoding="utf-8", errors="ignore")[:20000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"]).delimiter
    except Exception:
        return default

sep = detect_delimiter(src)

# Leer SIN encabezado
try:
    df = pd.read_csv(src, sep=sep, encoding="utf-8", header=None)
except UnicodeDecodeError:
    df = pd.read_csv(src, sep=sep, encoding="latin-1", header=None)

# Si todo vino en una sola columna, expandirla
if df.shape[1] == 1:
    df = df.iloc[:, 0].astype(str).str.split(sep, expand=True)

# Enmascarar la columna de nombres (todas las filas)
if 0 <= NAME_COL_IDX < df.shape[1]:
    df.iloc[:, NAME_COL_IDX] = MASK_VALUE

# Guardar SIN encabezado, con BOM para Excel
df.to_csv(dst, index=False, header=False, sep=sep, encoding="utf-8-sig")

print(f"Separador: {sep!r}")
print("Copia con privacidad:", dst)
