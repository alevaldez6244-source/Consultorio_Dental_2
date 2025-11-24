import pandas as pd
from pathlib import Path

# 1) Ruta de entrada (ajústala si tu archivo está en otro lugar o es .xlsx)
src = Path("pacientes_final.csv")  # o Path("pacientes_final.xlsx")

# 2) Nombre exacto de la columna a eliminar (si lo sabes). Ejemplo: "Pacientes"
COLUMNA_OBJETIVO = None  # ej. "Pacientes"

# 3) Conjunto de nombres comunes que intentaremos detectar si no especificas la columna
nombres_comunes = {
    "paciente", "pacientes", "nombre_paciente", "id_paciente",
    "historia_clinica", "hc_paciente"
}

# --- Cargar dataset (CSV o Excel) ---
if src.suffix.lower() == ".csv":
    df = pd.read_csv(src)
elif src.suffix.lower() in {".xlsx", ".xls"}:
    df = pd.read_excel(src)
else:
    raise ValueError(f"Formato no soportado: {src.suffix}")

cols = list(df.columns)

# --- Determinar columnas a eliminar ---
cols_bajar = set()

# a) Si especificaste el nombre exacto
if COLUMNA_OBJETIVO:
    # Coincidencia exacta (insensible a mayúsculas)
    match = [c for c in cols if c.lower() == COLUMNA_OBJETIVO.lower()]
    if not match:
        # Coincidencia por contiene (p. ej., "Pacientes (2025)")
        match = [c for c in cols if COLUMNA_OBJETIVO.lower() in c.lower()]
    cols_bajar.update(match)

# b) Detección automática por nombres comunes y patrones
if not cols_bajar:
    # exactos
    cols_bajar.update([c for c in cols if c.lower() in nombres_comunes])
    # patrones típicos
    cols_bajar.update([c for c in cols if "pacient" in c.lower()])  # paciente/pacientes/…
    # (opcional) agrega más reglas si lo necesitas

# --- Eliminar y guardar ---
df_out = df.drop(columns=list(cols_bajar), errors="ignore")

out_path = Path("pacientes_final_privacidad_clientes.xlsx")
df_out.to_excel(out_path, index=False)

print(f"Columnas eliminadas: {sorted(cols_bajar)}")
print(f"Archivo guardado en: {out_path.resolve()}")
