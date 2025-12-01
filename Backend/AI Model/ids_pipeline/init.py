# __init__.py
# Package initialization

__version__ = "1.0.0"
__author__ = "IDS Pipeline Team"

from .main import main
from .model_loader import ModelLoader
from .feature_extractor import FeatureExtractor
from .xai_explainer import XAIExplainer
from .packet_storage import PacketStorage, Packet
from .detector import Detector
from .api_server import APIServer
from .pipeline_manager import PipelineManager

__all__ = [
    'main',
    'ModelLoader',
    'FeatureExtractor',
    'XAIExplainer',
    'PacketStorage',
    'Packet',
    'Detector',
    'APIServer',
    'PipelineManager'
]