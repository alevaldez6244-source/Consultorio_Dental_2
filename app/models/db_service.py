# Archivo: app/models/db_service.py
import sqlite3
from app.config import DATA_DIR
from app.utils.helpers import toast

class DBService:
    """Maneja la persistencia de reportes en SQLite."""
    
    def __init__(self):
        self.db_path = DATA_DIR / "consultorio.db"

    def save_prediction_report(self, usuario, fecha_creacion, nombre, tipo, csv_path, parametros, descripcion):
        """Inserta el registro del reporte en la BD."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            query = """
                INSERT INTO predicciones
                (usuario, fecha_creacion, nombre, tipo, archivo_csv, parametros, descripcion)
                VALUES (?,?,?,?,?,?,?)
            """
            cur.execute(query, (usuario, fecha_creacion, nombre, tipo, 
                                str(csv_path), parametros, descripcion))
            
            conn.commit()
            conn.close()
            toast("Registro guardado en base de datos ✔️", "💾")
            return True
        except Exception as e:
            toast(f"Error al guardar en BD: {e}", "❌")
            return False