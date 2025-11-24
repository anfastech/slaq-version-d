from django.conf import settings

def get_supabase_config():
    return {
        'url': settings.SUPABASE_URL,
        'key': settings.SUPABASE_ANON_KEY,
        'service_role_key': settings.SUPABASE_SERVICE_ROLE_KEY,
    }

SUPABASE_CONFIG = get_supabase_config()