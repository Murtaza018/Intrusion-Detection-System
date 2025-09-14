# Keeps audit trail of system activity (events, scans, rule checks).

from sqlalchemy import Column, Integer, String
from ..dbConnection import Base

class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=False)
    level = Column(String, nullable=False)  # info, warning, error
    message = Column(String, nullable=False)
