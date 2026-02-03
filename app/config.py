# Archivo: app/config.py
import sys
from pathlib import Path

# Definimos la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Rutas de Directorios
DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models" / "models"
REPORTS_DIR = MODELS_DIR / "reportes"

# Asegurar que existan
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración Visual Global
PAGE_CONFIG = {
    "page_title": "🦷 Consultorio Dental",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}