git pull
PerceptPlatformEnv/bin/python3 manage.py makemigrations
PerceptPlatformEnv/bin/python3 manage.py migrate
PerceptPlatformEnv/bin/python3 manage.py collectstatic
systemctl reload apache2
