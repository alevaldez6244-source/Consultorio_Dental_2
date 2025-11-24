# src/list_users.py
import sqlite3
from pathlib import Path

# Ruta a la base de datos del consultorio
db_path = Path(__file__).parents[1] / "data" / "consultorio.db"

def list_users():
    """
    Lista todos los usuarios registrados en la tabla 'usuarios'.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM usuarios;")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No hay usuarios registrados en la base de datos.")
    else:
        print("Usuarios registrados:")
        for uid, user in rows:
            print(f" • {uid}: {user}")

if __name__ == "__main__":
    list_users()
