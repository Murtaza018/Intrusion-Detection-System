from sqlalchemy import Column, Integer, String
from ..dbConnection import Base

class Traffic(Base):
    __tablename__ = "traffic"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=False)
    source_ip = Column(String, nullable=False)
    destination_ip = Column(String, nullable=False)
    protocol = Column(String, nullable=True)
    packet_size = Column(Integer, nullable=True)
    payload = Column(String, nullable=True)  # or store only summary
