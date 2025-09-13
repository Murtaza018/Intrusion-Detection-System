from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parents[2]
dotenv_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=dotenv_path)

DATABASE_URL = os.getenv("DATABASE_URL")

#creates engine and session(holds the pool of connections)
engine = create_engine(DATABASE_URL)
#creates a local session to hold all the db objects temporarily before making changes permananet
#autocommit=False: changes are not saved until explicitly committed
#autoflush=False: changes are not automatically flushed to the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#base class for all the models
Base = declarative_base()

# Dependency for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
