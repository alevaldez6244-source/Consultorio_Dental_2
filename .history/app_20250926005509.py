# app.py
import os, sys, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/"src"), str(ROOT/"app")]

DB_PATH = Path(os.getenv("DB_PATH", ROOT/"data"/"app.db"))
ADMIN_USER = os.getenv("APP_ADMIN_USER", "consultorio")
ADMIN_PASS = os.getenv("APP_ADMIN_PASSWORD", "consultorio")

def bootstrap_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists() and DB_PATH.stat().st_size > 0:
        return
    # 1) crear tablas
    import runpy as _runpy
    try:
        import src.db as db
        if hasattr(db, "init_db"): db.init_db()
        elif hasattr(db, "create_db"): db.create_db()
        elif hasattr(db, "setup"): db.setup()
        else: _runpy.run_module("src.db", run_name="__main__")
    except Exception as e:
        print("[DB] Error creando tablas:", e)
    # 2) crear usuario
    try:
        import src.add_user as au
        if hasattr(au, "add_user"):
            try: au.add_user(ADMIN_USER, ADMIN_PASS)
            except TypeError: au.add_user()
        else:
            _runpy.run_module("src.add_user", run_name="__main__")
    except Exception as e:
        print("[DB] Error creando usuario:", e)

def run_streamlit():
    for p in [
        ROOT/"app"/"sistema.web_final.py",
        ROOT/"app"/"sistema_Web_final.py",
        ROOT/"app"/"sistema_web_final.py",
        ROOT/"app"/"sistema_web.py",
    ]:
        if p.exists():
            runpy.run_path(str(p), run_name="__main__")
            return
    raise FileNotFoundError("No se encontró tu script de Streamlit en /app")

if __name__ == "__main__":
    bootstrap_db()
    run_streamlit()
