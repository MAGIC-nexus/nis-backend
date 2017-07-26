# -*- coding: utf-8 -*-

"""
To publish using Apache:
* install mod_wsgi
* create a file with the following contents (possible to change the
  ip and port) named "pozo.magic.itccanarias.org.conf" (adapt it) at "/etc/apache2/sites-available"
<VirtualHost 10.141.188.67:8080>
    ServerAdmin admin@magic.itccanarias.org
    ServerName magic.itccanarias.org
    ServerAlias www.magic.itccanarias.org
    DocumentRoot /var/www/swmagic
    ErrorLog ${APACHE_LOG_DIR}/error.magic.log
    CustomLog ${APACHE_LOG_DIR}/access.magic.log combined

    WSGIDaemonProcess magic_pozo
    WSGIScriptAlias / /var/www/swmagic/nis/nis.wsgi
    <Directory /var/www/swmagic>
        WSGIProcessGroup magic_pozo
        WSGIApplicationGroup %{GLOBAL}
        Order deny,allow
        Allow from all
    </Directory>

</VirtualHost>

* cd "/etc/apache2"
* create a symbolic link "ln -s sites-available/pozo.magic.itccanarias.org.conf sites-enabled/pozo.magic.itccanarias.org.conf
* declare the used port in the file "ports.conf"
* sudo service apache2 restart

"""
import io
import json

import sqlalchemy.schema
from flask import abort, Response, url_for, request, send_from_directory, render_template
from flask_httpauth import HTTPBasicAuth
from flaskext.auth import Auth

from flask.helpers import get_root_path, safe_join
from werkzeug.debug import get_current_traceback

from backend.restful_service import app

UPLOAD_FOLDER = '/tmp/'

JSON_INDENT = 4
ENSURE_ASCII = False

auth = HTTPBasicAuth()
auth2 = Auth(app)

@auth.hash_password
def hash_pw(password):
    return md5(password).hexdigest()

# # Initialize DATABASE
# if 'DB_CONNECTION_STRING' in app.config:
#     db_connection_string = app.config['DB_CONNECTION_STRING']
#
#     engine = sqlalchemy.create_engine(db_connection_string, echo=True)
#     # global DBSession # global DBSession registry to get the scoped_session
#     DBSession.configure(bind=engine)  # reconfigure the sessionmaker used by this scoped_session
#     tables = ORMBase.metadata.tables
#     connection = engine.connect()
#     table_existence = [engine.dialect.has_table(connection, tables[t].name) for t in tables]
#     connection.close()
#     if False in table_existence:
#         ORMBase.metadata.bind = engine
#         ORMBase.metadata.create_all()
# else:
#     print("No database connection defined (DB_CONNECTION_STRING)!")


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    from datetime import datetime
    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial
    raise TypeError("Type not serializable")


"""
Publish hello world
Behind OAuth
"""


@app.route('/test', methods=['GET'])
@auth.login_required
def hello():
    return Response("Hello world!!!", status=200)


if __name__ == '__main__':
    # "cannot connect to X server" error when remote debugging?
    # Execute "Xvfb :99 -ac -noreset" in the remote server and uncomment the following line
    # os.environ["DISPLAY"] = ":99"
    app.run(host='0.0.0.0', debug=True, threaded=True)  # Default port, 5000
