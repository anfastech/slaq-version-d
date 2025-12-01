import os
import logging
from django.core.files.storage import Storage
from django.conf import settings

from supabase import create_client
from .supabase_config import get_supabase_config

logger = logging.getLogger(__name__)

supabase_initialized = False
supabase_client = None


def init_supabase():
    global supabase_initialized, supabase_client

    if settings.ENVIRONMENT != 'production':
        return

    if supabase_initialized:
        return

    cfg = get_supabase_config()
    supabase_client = create_client(cfg['url'], cfg['key'])
    supabase_initialized = True


class SupabaseStorage(Storage):
    def __init__(self):
        init_supabase()
        if not supabase_client:
            raise Exception("Supabase client not initialized")
        self.bucket = supabase_client.storage

    def _save(self, name, content):
        try:
            # Normalize path separators to forward slashes for Supabase
            name = name.replace('\\', '/')

            content_bytes = content.read()
            content.seek(0)

            self.bucket.from_(settings.SUPABASE_BUCKET_NAME).upload(
                path=name,
                file=content_bytes,
                file_options={
                    "content-type": content.content_type if hasattr(content, "content_type") else "application/octet-stream"
                }
            )
            return name
        except Exception as e:
            logger.error(f"upload error {name}: {e}")
            raise

    def _open(self, name, mode="rb"):
        try:
            # Normalize path separators
            name = name.replace('\\', '/')
            data = self.bucket.from_(settings.SUPABASE_BUCKET_NAME).download(name)
            from io import BytesIO
            return BytesIO(data)
        except Exception as e:
            logger.error(f"download error {name}: {e}")
            raise

    def delete(self, name):
        try:
            # Normalize path separators
            name = name.replace('\\', '/')
            self.bucket.from_(settings.SUPABASE_BUCKET_NAME).remove([name])
        except Exception as e:
            logger.error(f"delete error {name}: {e}")
            raise

    def exists(self, name):
        try:
            # Normalize path separators
            name = name.replace('\\', '/')
            folder = os.path.dirname(name) or ""
            base = os.path.basename(name)
            items = self.bucket.from_(settings.SUPABASE_BUCKET_NAME).list(path=folder)
            return any(i["name"] == base for i in items)
        except:
            return False

    def url(self, name):
        try:
            # Normalize path separators
            name = name.replace('\\', '/')
            url = self.bucket.from_(settings.SUPABASE_BUCKET_NAME).get_public_url(name)
            return url
        except Exception as e:
            logger.error(f"url error {name}: {e}")
            return ""