# src/add_user.py
import sqlite3
import bcrypt
from pathlib import Path
import sys

db_path = Path(__file__).parents[1] / "data" / "consultorio.db"


def hash_password(password: str) -> str:
    """
    Genera un hash de la contraseña usando bcrypt.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def add_user(username: str, password: str, rol_id: int = 2):
    """
    Inserta un nuevo usuario en la tabla 'usuarios'.
    Por defecto, rol_id=2 (p. ej. rol estándar).
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    cursor = conn.cursor()
    pw_hash = hash_password(password)
    cursor.execute(
        "INSERT INTO usuarios(username, password_hash, rol_id) VALUES(?, ?, ?)",
        (username, pw_hash, rol_id)
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python add_user.py <username> <password> [rol_id]")
        sys.exit(1)
    usuario = sys.argv[1]
    contrasenia = sys.argv[2]
    rol = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    add_user(usuario, contrasenia, rol)
    print(f"✔ Usuario '{usuario}' creado con rol_id {rol}")
