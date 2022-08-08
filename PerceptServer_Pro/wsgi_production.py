"""
WSGI config for PerceptServer_Pro project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ["PERCEPT_DIR"] = "/home/jcagle/Storage/fixel_academy/BRAVO_SSR/modules/PerceptProcessingModules"
os.environ["STATIC_ROOT"] = "/home/jcagle/Storage/fixel_academy/BRAVO_SSR/PerceptServer_Pro/static"
os.environ["SECRET_KEY"] = "cXPzwyfdVdlwUApVNYliiXhujtfJYsAvYFXGDHatOiwfGZQlXsCPvXnMDnqBiiBh"
os.environ["DATASERVER_PATH"] = "/home/jcagle/Storage/Data/Percept_DataServer_2"
os.environ["SERVER_ADDRESS"] = "0.0.0.0"
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PerceptServer_Pro.settings')

application = get_wsgi_application()
