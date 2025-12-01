# diagnosis/tasks.py
from celery import shared_task
from django.utils import timezone
from django.conf import settings
import logging
import librosa
import tempfile
import os

from .models import AudioRecording, AnalysisResult
from .ai_engine.model_loader import get_stutter_detector, log_model_cache_info

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_audio_recording(self, recording_id):
    """
    Async task to process audio recording (MVP Simplified)
    Following SLAQ AI Workflow:
    1. Audio Input
    2. AI Diagnosis (Articulation analysis)
    3. Results Storage
    """
    
    try:
        logger.info(f"🎯 Processing recording {recording_id}")
        
        # Get recording
        recording = AudioRecording.objects.get(id=recording_id)
        recording.status = 'processing'
        recording.save()

        # Download audio file to temp location for processing
        temp_audio_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(recording.audio_file.name)[1]) as temp_file:
            temp_file.write(recording.audio_file.read())
            temp_audio_path = temp_file.name
        audio_path = temp_audio_path
        
        # Calculate duration
        try:
            duration = librosa.get_duration(path=audio_path)
            recording.duration_seconds = round(duration, 2)
            recording.save()
            logger.info(f"📏 Audio duration: {duration:.2f} seconds")
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate duration: {e}")
        
        # Load AI detector and analyze audio via external API
        logger.info(f"🤖 Loading AI detector (external API mode)...")
        log_model_cache_info()  # Log API mode info
        detector = get_stutter_detector()
        
        logger.info(f"🎵 Analyzing audio via external ML API...")
        logger.info(f"🎵 Audio path: {audio_path}")
        logger.info(f"🎵 Audio file exists: {os.path.exists(audio_path)}")
        analysis_data = detector.analyze_audio(audio_path)
        
        # Save analysis results
        analysis = AnalysisResult.objects.create(
            recording=recording,
            actual_transcript=analysis_data['actual_transcript'],
            target_transcript=analysis_data['target_transcript'],
            mismatched_chars=analysis_data['mismatched_chars'],
            mismatch_percentage=analysis_data['mismatch_percentage'],
            ctc_loss_score=analysis_data['ctc_loss_score'],
            stutter_timestamps=analysis_data['stutter_timestamps'],
            total_stutter_duration=analysis_data['total_stutter_duration'],
            stutter_frequency=analysis_data['stutter_frequency'],
            severity=analysis_data['severity'],
            confidence_score=analysis_data['confidence_score'],
            analysis_duration_seconds=analysis_data['analysis_duration_seconds'],
            model_version=analysis_data['model_version']
        )
        
        # Update recording status
        recording.status = 'completed'
        recording.processed_at = timezone.now()
        recording.save()
        
        logger.info(f"✅ Recording {recording_id} processed successfully")
        
        return {
            'recording_id': recording_id,
            'analysis_id': analysis.id,
            'severity': analysis.severity,
            'mismatch_percentage': analysis.mismatch_percentage
        }

    except AudioRecording.DoesNotExist:
        logger.error(f"❌ Recording {recording_id} not found")
        raise

    except Exception as e:
        logger.error(f"❌ Processing failed for recording {recording_id}: {e}")

        try:
            recording = AudioRecording.objects.get(id=recording_id)
            recording.status = 'failed'
            recording.error_message = str(e)
            recording.save()
        except:
            pass

        # Retry task
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))

    finally:
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


# slaq_project/celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'slaq_project.settings')

app = Celery('slaq_project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')