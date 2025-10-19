
import os, sys, time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

MIGRATIONS_SQL = "migrations.sql"

def init_db():
    with engine.connect() as conn:
        with open(MIGRATIONS_SQL, "r") as f:
            sql = f.read()
        # execute many statements
        for stmt in sql.split(";"):
            s = stmt.strip()
            if not s:
                continue
            conn.execute(text(s))
        conn.commit()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Run migrations.sql")
    args = parser.parse_args()
    if args.init:
        for i in range(10):
            try:
                init_db()
                print("DB initialized.")
                return
            except Exception as e:
                print("DB init retrying...", e)
                time.sleep(2)
        print("DB init failed", file=sys.stderr)

if __name__ == "__main__":
    main()
