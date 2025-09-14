# Stores app users (admins, security analysts, normal users).
# Handles login/authentication.
# Each user can set rules, check alerts, and provide feedback.

from sqlalchemy import Column, Integer, String
from dbConnection.dbConnection import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="analyst")  
    # roles: admin, analyst, user
