# diagnosis/ai_engine/detect_stuttering.py
"""
Stutter detection using external ML API endpoint
All analysis is performed by the external API hosted on HuggingFace
"""
import logging
import os
import time
import requests
from typing import Dict

logger = logging.getLogger(__name__)


class StutterDetector:
    """
    Stutter detection using external ML API
    API endpoint: https://anfastech-slaq-version-d-ai-test-engine.hf.space/analyze
    """
    
    def __init__(self):
        """Initialize detector - no local models needed"""
        logger.info("🔄 Initializing StutterDetector (API-only mode)")
        self.api_url = "https://anfastech-slaq-version-d-ai-test-engine.hf.space/analyze"
        logger.info("✅ StutterDetector initialized (using external API)")
    
    def analyze_audio(self, audio_file_path: str, proper_transcript: str = "") -> Dict:
        """
        Analyze audio using external ML API endpoint
        
        Args:
            audio_file_path: Path to audio file
            proper_transcript: Optional expected transcript (if available)
        
        Returns:
            Dictionary with complete analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting API analysis for: {audio_file_path}")
            logger.info(f"🔍 Transcript provided: {bool(proper_transcript)}")
            
            # Verify file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Get file info
            file_size = os.path.getsize(audio_file_path)
            logger.info(f"📋 API URL: {self.api_url}")
            logger.info(f"📋 Transcript value: '{proper_transcript if proper_transcript else ''}'")
            logger.info(f"📋 File size: {file_size} bytes")
            
            # Open audio file and prepare for upload
            with open(audio_file_path, "rb") as f:
                files = {"audio": f}
                data = {"transcript": proper_transcript if proper_transcript else ""}
                
                logger.info(f"📤 Sending POST request to {self.api_url}")
                logger.info(f"📤 Files dict keys: {list(files.keys())}")
                logger.info(f"📤 Data dict: {data}")
                
                try:
                    response = requests.post(self.api_url, files=files, data=data, timeout=300)
                    logger.info(f"📥 Response status code: {response.status_code}")
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    logger.info(f"✅ API response received: {type(result)}")
                    logger.info(f"✅ API response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                except requests.exceptions.RequestException as req_err:
                    logger.error(f"❌ Request exception details: {type(req_err).__name__}: {str(req_err)}")
                    if hasattr(req_err, 'response') and req_err.response is not None:
                        logger.error(f"❌ Response status: {req_err.response.status_code}")
                        logger.error(f"❌ Response text: {req_err.response.text[:500]}")
                    raise
            
            # Calculate analysis duration
            analysis_duration = time.time() - start_time
            
            # Ensure result has all required fields with defaults if missing
            formatted_result = {
                'actual_transcript': result.get('actual_transcript', ''),
                'target_transcript': result.get('target_transcript', proper_transcript.upper() if proper_transcript else ''),
                'mismatched_chars': result.get('mismatched_chars', []),
                'mismatch_percentage': result.get('mismatch_percentage', 0.0),
                'ctc_loss_score': result.get('ctc_loss_score', 0.0),
                'stutter_timestamps': result.get('stutter_timestamps', []),
                'total_stutter_duration': result.get('total_stutter_duration', 0.0),
                'stutter_frequency': result.get('stutter_frequency', 0.0),
                'severity': result.get('severity', 'none'),
                'confidence_score': result.get('confidence_score', 0.0),
                'analysis_duration_seconds': round(analysis_duration, 2),
                'model_version': result.get('model_version', 'external-api'),
            }
            
            logger.info(f"✅ API analysis complete in {analysis_duration:.2f}s")
            return formatted_result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ API analysis failed: {e}")
            raise
