# src/show_db_schema.py
import sqlite3
from pathlib import Path


db_path = Path(__file__).parents[1] / "data" / "consultorio.db"

def show_schema():
    """
    Muestra los nombres de tablas y campos (columnas) en la base de datos.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        print("No hay tablas en la base de datos.")
    else:
        for (table_name,) in tables:
            print(f"\nTabla: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name});")
            cols = cursor.fetchall()
            if not cols:
                print("  (Sin columnas)")
            else:
                print("  Campos:")
                for col in cols:
                    print(f"   - {col[1]} ({col[2]})")
    conn.close()

if __name__ == "__main__":
    show_schema()
