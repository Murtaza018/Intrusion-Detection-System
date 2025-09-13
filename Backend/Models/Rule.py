from sqlalchemy import Column, Integer, String
from ..dbConnection import Base

class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    pattern = Column(String, nullable=False)   # regex, signature, ML tag
    description = Column(String, nullable=True)
    severity = Column(String, default="medium")
