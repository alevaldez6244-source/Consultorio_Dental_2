from pathlib import Path
import sqlite3
import bcrypt

# Conexión a la base de datos del consultorio
db_path = Path(__file__).parents[1] / "data" / "consultorio.db"
conn = sqlite3.connect(str(db_path), check_same_thread=False)

def check_credentials(username: str, password: str) -> bool:
    """
    Verifica que el 'username' exista en la tabla 'usuarios' y que la contraseña coincida con el hash.
    Devuelve True si las credenciales son válidas, False en caso contrario.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password_hash FROM usuarios WHERE username = ?", (username,)
    )
    row = cursor.fetchone()
    if row is None:
        return False
    stored_hash = row[0]
    return bcrypt.checkpw(password.encode(), stored_hash.encode())