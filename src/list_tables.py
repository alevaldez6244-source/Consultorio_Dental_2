# src/list_tables.py
import sqlite3
from pathlib import Path

# Ruta a la base de datos del consultorio
db_path = Path(__file__).parents[1] / "data" / "consultorio.db"

def list_tables():
    """
    Lista todas las tablas presentes en la base de datos.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()

    print("Tablas en la base de datos:")
    for (tbl,) in tables:
        print(f" • {tbl}")

if __name__ == "__main__":
    list_tables()