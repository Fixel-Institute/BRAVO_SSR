"""
WSGI config for PerceptAnalysis project for Windows Platform.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

SERVER_DIRECTORY = "C:/Users/Username/Documents/GitHub/BRAVO_SSR/"

activate_this = SERVER_DIRECTORY + "PerceptPlatformEnv/Scripts/activate_this.py"
exec(open(activate_this).read(),dict(__file__=activate_this))

import os, sys

sys.path.append(SERVER_DIRECTORY)
sys.path.append(SERVER_DIRECTORY + "PerceptDashboard")

from django.core.wsgi import get_wsgi_application

os.environ["PERCEPT_DIR"] = SERVER_DIRECTORY + "modules/PerceptProcessingModules"
os.environ["STATIC_ROOT"] = SERVER_DIRECTORY + "static"
os.environ["SECRET_KEY"] = "ThisIsNotSecretReplaceWithNewSecretForProduction"
os.environ["SERVER_ADDRESS"] = "0.0.0.0"
os.environ["DATASERVER_PATH"] = SERVER_DIRECTORY + "Storage/"
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PerceptAnalysis.settings')

application = get_wsgi_application()
