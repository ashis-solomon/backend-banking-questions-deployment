"""
Django settings for core project.

Generated by 'django-admin startproject' using Django 4.1.2.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.1/ref/settings/
"""

from pathlib import Path
from decouple import config
import dj_database_url
import os
import nltk

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
# SECRET_KEY = 'django-insecure-!^-1np4s)nm1ah)om$0#0a@xurvqbugqly(oymh6kw3qk3=&f9'

SECRET_KEY = config('SECRET_KEY', default='django-insecure-!^-1np4s)nm1ah)om$0#0a@xurvqbugqly(oymh6kw3qk3=&f9')

# SECURITY WARNING: don't run with debug turned on in production!
# DEBUG = True

DEBUG = config('DEBUG', default=False, cast=bool)

# ALLOWED_HOSTS = ["*"]
# CORS_ORIGIN_ALLOW_ALL = True

# ALLOWED_HOSTS = ['.onrender.com', '.vercel.app']

# CORS_ALLOWED_ORIGINS = [
#     'https://frontend-banking-questions-deployment.vercel.app',
# ]

ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=lambda v: [host.strip() for host in v.split(',')])
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', cast=lambda v: [origin.strip() for origin in v.split(',')])

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'corsheaders',
    'rest_framework',
    'drf_spectacular',
    # 'drf_yasg',

    'api.apps.ApiConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }

# DATABASES = {
#     'default': dj_database_url.parse(os.environ.get("DATABASE_URL")),
# }

DATABASES = {
    'default': config('DATABASE_URL', cast=dj_database_url.parse),
}

# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

# TIME_ZONE = 'UTC'
TIME_ZONE = config('TIME_ZONE', default='UTC')

USE_I18N = True

USE_TZ = True

# SECURE_HSTS_SECONDS = 3600
# SECURE_SSL_REDIRECT = True
# SESSION_COOKIE_SECURE = True
# CSRF_COOKIE_SECURE = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = 'static/'
MEDIA_URL = 'media/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# User Defined

MODELS = {
    'adaboost': 'api/models/oversampled_adaboost_classifier.pkl',
    'svm': 'api/models/oversampled_svm_classifier.pkl',
    'naive_bayes': 'api/models/oversampled_naive bayes_classifier.pkl',
    'random_forest': 'api/models/oversampled_random forest_classifier.pkl',
    'gradient_boosting': 'api/models/oversampled_gradient boosting_classifier.pkl'
}


REST_FRAMEWORK = {
    # YOUR SETTINGS
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}

SPECTACULAR_SETTINGS = {
    'TITLE': 'Banking Questions API',
    'DESCRIPTION': 'Enhance online customer service in the banking sector by automating the initial filtering process of customer queries.',
    'VERSION': '1.0.0',
    'SERVE_INCLUDE_SCHEMA': False,
    # OTHER SETTINGS
}


# Determine the path for the NLTK resources within the 'packages' folder
nltk_data_path = os.path.join(BASE_DIR, 'api', 'packages', 'nltk_data')

# Set the custom path for NLTK resources
nltk.data.path.append(nltk_data_path)

# Check if the 'nltk_data' directory exists, and create it if necessary
if not os.path.isdir(nltk_data_path):
    os.makedirs(nltk_data_path)

# Download the required resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

NLTK_DATA = nltk_data_path