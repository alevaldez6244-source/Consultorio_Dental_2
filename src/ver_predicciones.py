import sqlite3
from pathlib import Path
import pandas as pd

# Ruta correcta, igual que en tu base de datos
db_path = Path(__file__).parents[1] / "data" / "consultorio.db"

def ver_predicciones():
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT id, usuario, fecha_creacion, nombre, tipo, archivo_csv, parametros, descripcion
        FROM predicciones
        ORDER BY fecha_creacion DESC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No hay predicciones guardadas en la base de datos.")
    else:
        print("\nPREDICCIONES REGISTRADAS:")
        print(df[["id", "usuario", "fecha_creacion", "nombre", "tipo", "archivo_csv", "descripcion"]].to_string(index=False))
        print("\n¿Ver una previsualización de algún CSV? Ingresa el número de ID (o Enter para salir):")
        sel = input().strip()
        if sel and sel.isdigit():
            row = df[df["id"] == int(sel)]
            if not row.empty:
                archivo_csv = row.iloc[0]["archivo_csv"]
                if Path(archivo_csv).exists():
                    df_csv = pd.read_csv(archivo_csv)
                    print(f"\nVista previa de {archivo_csv}:")
                    print(df_csv.head(10))
                else:
                    print(f"No se encontró el archivo {archivo_csv}")
        print("\nFin.")

if __name__ == "__main__":
    ver_predicciones()
