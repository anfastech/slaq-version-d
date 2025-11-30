# diagnosis/ai_engine/model_loader.py
"""Singleton pattern for model loading"""
import logging
import os
from .detect_stuttering import StutterDetector

logger = logging.getLogger(__name__)
_detector_instance = None

def get_stutter_detector():
    """Get or create singleton StutterDetector instance"""
    global _detector_instance
    if _detector_instance is None:
        logger.info("🤖 Initializing StutterDetector singleton instance...")
        _detector_instance = StutterDetector()
        logger.info("✅ StutterDetector singleton created successfully")
    else:
        logger.debug("🔄 Using existing StutterDetector singleton instance")
    return _detector_instance

def log_model_cache_info():
    """Log information about model cache location and status"""
    try:
        # Check Hugging Face cache location
        hf_cache = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
        if not hf_cache:
            # Default cache location
            home = os.path.expanduser('~')
            hf_cache = os.path.join(home, '.cache', 'huggingface')

        logger.info(f"📂 Hugging Face cache location: {hf_cache}")

        # Check if cache exists
        if os.path.exists(hf_cache):
            # Get cache size (rough estimate)
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(hf_cache):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass

            cache_size_mb = total_size / (1024 * 1024)
            logger.info(f"💾 Cache size: {cache_size_mb:.1f} MB")
        else:
            logger.warning("⚠️ Hugging Face cache directory does not exist yet")

    except Exception as e:
        logger.error(f"❌ Error checking cache info: {e}")
