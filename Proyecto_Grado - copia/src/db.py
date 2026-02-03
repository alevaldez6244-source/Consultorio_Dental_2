import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parents[1] / "data" / "consultorio.db"

def init_db():
    """
    Inicializa la base de datos SQLite y crea las tablas necesarias para el consultorio dental.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT UNIQUE NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            rol_id INTEGER NOT NULL,
            FOREIGN KEY (rol_id) REFERENCES roles(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS areas (
            id INTEGER PRIMARY KEY,
            nombre TEXT NOT NULL UNIQUE
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odontologos (
            id INTEGER PRIMARY KEY,
            nombre TEXT NOT NULL,
            apellido TEXT NOT NULL,
            telefono TEXT,
            email TEXT,
            area_id INTEGER,
            FOREIGN KEY (area_id) REFERENCES areas(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY,
            nombre_completo TEXT NOT NULL,
            edad INTEGER,
            genero TEXT,
            fecha_registro TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tratamientos (
            id INTEGER PRIMARY KEY,
            nombre TEXT NOT NULL,
            area_id INTEGER,
            costo REAL,
            FOREIGN KEY (area_id) REFERENCES areas(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS citas (
            id INTEGER PRIMARY KEY,
            paciente_id INTEGER NOT NULL,
            odontologo_id INTEGER NOT NULL,
            fecha_hora TEXT NOT NULL,
            estado TEXT NOT NULL,
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id),
            FOREIGN KEY (odontologo_id) REFERENCES odontologos(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS atenciones (
            id INTEGER PRIMARY KEY,
            cita_id INTEGER NOT NULL,
            tratamiento_id INTEGER NOT NULL,
            fecha_atencion TEXT NOT NULL,
            notas TEXT,
            FOREIGN KEY (cita_id) REFERENCES citas(id),
            FOREIGN KEY (tratamiento_id) REFERENCES tratamientos(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS facturas (
            id INTEGER PRIMARY KEY,
            atencion_id INTEGER NOT NULL,
            fecha_emision TEXT NOT NULL,
            total REAL NOT NULL,
            FOREIGN KEY (atencion_id) REFERENCES atenciones(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pagos (
            id INTEGER PRIMARY KEY,
            factura_id INTEGER NOT NULL,
            fecha_pago TEXT NOT NULL,
            monto REAL NOT NULL,
            metodo TEXT NOT NULL,
            FOREIGN KEY (factura_id) REFERENCES facturas(id)
        );
    """)
    cursor.execute("DROP TABLE IF EXISTS predicciones;")
    cursor.execute("""
    CREATE TABLE predicciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario TEXT,
        fecha_creacion TEXT,
        nombre TEXT,
        tipo TEXT,
        archivo_csv TEXT,
        parametros TEXT,
        descripcion TEXT
    );
""")


    conn.commit()
    return conn

if __name__ == "__main__":
    init_db()
    print("✔ Base de datos 'consultorio.db' creada/actualizada en data/")

    