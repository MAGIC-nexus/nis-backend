# -*- coding: utf-8 -*-

from flask import Flask
app = Flask(__name__)
app.debug = True
UPLOAD_FOLDER = '/tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Initialize configuration
try:
    app.config.from_envvar('MAGIC_SERVICE_CONFIG_FILE')
except Exception:
    print("MAGIC_SERVICE_CONFIG_FILE environment variable not defined!")
