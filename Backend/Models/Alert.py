from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from ..dbConnection import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    description = Column(String, nullable=False)
    source_ip = Column(String, nullable=False)
    destination_ip = Column(String, nullable=False)
    protocol = Column(String, nullable=True)
    status = Column(String, default="open")
    rule_id = Column(Integer, ForeignKey("rules.id"))
    resolved_by = Column(Integer, ForeignKey("users.id"))

    rule = relationship("Rule")
    user = relationship("User")
