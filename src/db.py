import sqlite3
from pathlib import Path

# Rutas relativas al proyecto
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "consultorio.db"

def init_db():
    """
    Inicializa la base de datos SQLite con estructura optimizada
    para sistema Web + Machine Learning (Random Forest).
    """
    # Crear carpeta data si no existe
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()
    
    # Activar Claves Foráneas siempre
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. ROLES Y USUARIOS (Seguridad)
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

    # 2. ÁREAS (Fundamental para Random Forest)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS areas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL UNIQUE,
            descripcion TEXT
        );
    """)

    # 3. ODONTÓLOGOS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odontologos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            apellido TEXT NOT NULL,
            telefono TEXT,
            email TEXT,
            area_id INTEGER,
            FOREIGN KEY (area_id) REFERENCES areas(id)
        );
    """)

    # 4. PACIENTES (CORREGIDO: Usamos fecha_nacimiento y fecha_registro)
    # Nota: La IA necesita 'edad' (la calcularemos al exportar) y 'fecha_registro' (antigüedad).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            apellido TEXT NOT NULL,
            fecha_nacimiento DATE NOT NULL, 
            genero TEXT,
            telefono TEXT,
            email TEXT,
            direccion TEXT,
            fecha_registro DATE DEFAULT (DATE('now')) 
        );
    """)

    # 5. TRATAMIENTOS (CORREGIDO: Vinculado a Áreas)
    # Nota: Random Forest usa el area_id para predecir tendencias por especialidad.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tratamientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            descripcion TEXT,
            area_id INTEGER NOT NULL,
            costo REAL NOT NULL,
            FOREIGN KEY (area_id) REFERENCES areas(id)
        );
    """)

    # 6. CITAS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS citas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER NOT NULL,
            odontologo_id INTEGER,
            fecha_hora DATETIME NOT NULL,
            estado TEXT DEFAULT 'Pendiente', -- Pendiente, Confirmada, Cancelada, Completada
            motivo TEXT,
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id),
            FOREIGN KEY (odontologo_id) REFERENCES odontologos(id)
        );
    """)

    # 7. ATENCIONES (La tabla central para Ingresos)
    # Conecta una Cita con un Tratamiento Realizado.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS atenciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cita_id INTEGER, 
            paciente_id INTEGER NOT NULL, -- Redundancia útil para consultas rápidas del ML
            tratamiento_id INTEGER NOT NULL,
            fecha_atencion DATE NOT NULL,
            notas TEXT,
            FOREIGN KEY (cita_id) REFERENCES citas(id),
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id),
            FOREIGN KEY (tratamiento_id) REFERENCES tratamientos(id)
        );
    """)

    # 8. FACTURAS (Control Financiero)
    # Aquí agregamos estado_pago para que el CSV sepa si es "Pagado" o no.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS facturas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            atencion_id INTEGER NOT NULL,
            fecha_emision DATE DEFAULT (DATE('now')),
            total REAL NOT NULL,
            estado_pago TEXT DEFAULT 'Pendiente', -- 'Pagado', 'Pendiente', 'Parcial'
            FOREIGN KEY (atencion_id) REFERENCES atenciones(id)
        );
    """)

    # 9. PAGOS (Detalle de ingreso de dinero)
    # El 'metodo' es clave para tus gráficos (Efectivo, Tarjeta, QR).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pagos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factura_id INTEGER NOT NULL,
            fecha_pago DATE DEFAULT (DATE('now')),
            monto REAL NOT NULL,
            metodo TEXT NOT NULL, 
            comprobante TEXT,
            FOREIGN KEY (factura_id) REFERENCES facturas(id)
        );
    """)

    # 10. PREDICCIONES (Historial de reportes)
    cursor.execute("DROP TABLE IF EXISTS predicciones;")
    cursor.execute("""
        CREATE TABLE predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario TEXT,
            fecha_creacion DATETIME DEFAULT (DATETIME('now')),
            nombre TEXT,
            tipo TEXT,
            archivo_csv TEXT,
            parametros TEXT,
            descripcion TEXT
        );
    """)

    # --- DATOS INICIALES (SEMILLA) ---
    # Insertamos Áreas por defecto si está vacía, para que el sistema no arranque en blanco.
    cursor.execute("SELECT count(*) FROM roles")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO roles (nombre) VALUES ('Admin'), ('Odontologo'), ('Recepcion');")
        # Usuario admin/admin (Hash de prueba, en prod usar hash real)
        cursor.execute("INSERT INTO usuarios (username, password_hash, rol_id) VALUES ('admin', 'admin123', 1);")
        print("ℹ Usuarios iniciales creados.")

    conn.commit()
    conn.close()
    print(f"✔ Base de datos actualizada correctamente en: {DB_PATH}")

if __name__ == "__main__":
    init_db()