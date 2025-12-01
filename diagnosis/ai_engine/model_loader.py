# diagnosis/ai_engine/model_loader.py
"""Singleton pattern for detector loading"""
import logging
from .detect_stuttering import StutterDetector

logger = logging.getLogger(__name__)
_detector_instance = None

def get_stutter_detector():
    """Get or create singleton StutterDetector instance"""
    global _detector_instance
    if _detector_instance is None:
        logger.info("🤖 Initializing StutterDetector singleton instance (API mode)...")
        _detector_instance = StutterDetector()
        logger.info("✅ StutterDetector singleton created successfully")
    else:
        logger.debug("🔄 Using existing StutterDetector singleton instance")
    return _detector_instance

def log_model_cache_info():
    """No-op function - models are hosted externally on HuggingFace"""
    logger.info("🌐 Using external ML API - no local model cache needed")
