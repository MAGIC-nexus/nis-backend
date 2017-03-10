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
import json


import urllib

import sqlalchemy.orm
import sqlalchemy.schema
from flask import Flask, jsonify, abort, Response,  redirect, url_for, request, send_file, send_from_directory, render_template
from flask.helpers import get_root_path, safe_join
import werkzeug
from werkzeug.debug import get_current_traceback
from werkzeug.utils import secure_filename
from nis.model import DBSession, ORMBase, Diagram
import io
import magic  # Detect file type or content type
import xlrd
from nis.file_processing import process_file
from nis import app

UPLOAD_FOLDER = '/tmp/'

JSON_INDENT = 4
ENSURE_ASCII = False

# app = Flask(__name__)
# app.debug = True
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # Initialize configuration
# try:
#     app.config.from_envvar('MAGIC_SERVICE_CONFIG_FILE')
# except Exception:
#     print("MAGIC_SERVICE_CONFIG_FILE environment variable not defined!")
#     pass

# Initialize DATABASE
if 'DB_CONNECTION_STRING' in app.config:
    db_connection_string = app.config['DB_CONNECTION_STRING']

    engine = sqlalchemy.create_engine(db_connection_string, echo=True)
    # global DBSession # global DBSession registry to get the scoped_session
    DBSession.configure(bind=engine)  # reconfigure the sessionmaker used by this scoped_session
    tables = ORMBase.metadata.tables
    connection = engine.connect()
    table_existence = [engine.dialect.has_table(connection, tables[t].name) for t in tables]
    connection.close()
    if False in table_existence:
        ORMBase.metadata.bind = engine
        ORMBase.metadata.create_all()
else:
    print("No database connection defined (DB_CONNECTION_STRING)!")


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    from datetime import datetime
    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial
    raise TypeError("Type not serializable")


"""
Page for upload of document -> process -> download of result
List of sources
List of datasets per source
Get dataset
Put dataset?
List case studies
List detail case study
Graphical view of case study

Read Excel, interpret all the commands, produce internal data structures and generate new Excel
"""


@app.route('/magic-nis/file/<path:path>')
def send_static_files(path):
    safe = safe_join(app.config['UPLOAD_FOLDER'], path)
    # Read the file
    with open(safe, "rb") as f:
        b = f.read()
    # Elaborate response. jQuery fileDownload requires setting two cookies
    r = Response(b, status=200, mimetype="application/octet-stream")
    r.set_cookie("fileDownload", "true")
    r.set_cookie("path", "/")
    # file_type = magic.from_buffer(b, mime=True)
    # file_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet "
    # r.headers["Content-type"] = file_type
    r.headers["Content-Disposition"] = "attachment; filename=" + path
    r.headers["Content-Length"] = len(b)
    return r
    #return send_from_directory(app.config['UPLOAD_FOLDER'], path, as_attachment=True)


@app.route('/magic-nis/template_file/<id>')
def send_template(id):
    base = get_root_path("nis")
    if id == '1':
        return send_from_directory(base, 'templates/empty_template.xlt', as_attachment=True)
    elif id == '2':
        return send_from_directory(base, 'templates/test_based_on_template.xlsx', as_attachment=True)
    else:
        return None


@app.route('/magic-nis/file-transmuter', methods=['PUT', 'POST'])
def file_transmuter():
    # Receive a file, process, return elaborated file

    try:
        date_format = "%Y%m%d-%H%M%S"

        if len(request.files) > 0:
            for k in request.files:
                buffer = bytes(request.files[k].stream.getbuffer())
                break
        else:
            buffer = bytes(io.BytesIO(request.get_data()).getbuffer())

        import uuid
        filename = "mb_"+str(uuid.uuid4())

        file_type = magic.from_buffer(buffer, mime=True)

        extension = ""
        if file_type == "application/excel":
            extension = ".xlsx"
        elif file_type == "application/octet-stream":
            # Try opening as Excel
            try:
                data = xlrd.open_workbook(file_contents=buffer)
                extension = ".xlsx"
            except Exception:
                pass
        elif file_type == "text/x-python":
            extension = ".py"
        elif file_type == "image/png":
            extension = ".png"

        # Process the file
        buffer = process_file(buffer)

        # Write the resulting file
        extension2 = ".xls"
        with open(app.config['UPLOAD_FOLDER']+filename+extension2, "wb") as f:
            f.write(buffer)
        # Redirect to the resulting file
        return Response(url_for('send_static_files', path=filename+extension2))
        # return redirect(url_for('send_static_files', path=filename+extension))

        # resp = Response(buffer, mimetype=file_type)
        # resp.headers["Content-type"] = file_type
        # resp.headers["Content-Disposition"] = "attachment; filename=" + filename + extension

        # return resp
    except Exception as e:
        track = get_current_traceback(skip=1, show_hidden_frames=True, ignore_system_exceptions=False)
        track.log()
        abort(500)


@app.route('/magic-nis/magic-box.html')
def render_drag_drop_transformer():
    # TODO Pass it the target web service name, "/magic-nis/file-transmuter"
    return render_template('transceiver_jquery_upload_file.html')
    # return render_template('transceiver_jquery_html5_uploader.html')
    # return render_template('transceiver_dropzonejs.html')


# ------------------------

@app.route('/diagrams/', methods=['GET'])
def get_list_of_diagrams():
    session = DBSession()
    result = session.query(Diagram).all()
    m = []
    for d in result:
        m.append({"id": d.id, "page": d.page, "content": d.content})

    r = Response(json.dumps(m, default=json_serial,
                            sort_keys=True, indent=JSON_INDENT,
                            ensure_ascii=ENSURE_ASCII, separators=(',', ': ')),
                 mimetype="text/json")
    session.close()
    return r


@app.route('/diagrams/<string:page>', methods=['GET'])
def get_diagram_detail(page):
    """
    Obtain diagram JSON
    :param page:
    :return:
    """
    session = DBSession()
    d = session.query(Diagram).filter_by(page=page).first()
    if d:
        r = Response(json.dumps({"id": d.id, "page": d.page, "content": d.content}, default=json_serial,
                                sort_keys=True, indent=JSON_INDENT,
                                ensure_ascii=ENSURE_ASCII, separators=(',', ': ')),
                     mimetype="text/json")
    else:
        r = Response(json.dumps({"status": 404, "message": "'%s' not found" % page}, default=json_serial,
                                sort_keys=True, indent=JSON_INDENT,
                                ensure_ascii=ENSURE_ASCII, separators=(',', ': ')),
                     mimetype="text/json")
        r.status_code = 404
    session.close()
    return r


@app.route('/diagrams/<string:page>', methods=['PUT'])
def set_diagram_detail(page):
    """
    Save a diagram
    If it exists already, overwrite. ¿Also, keep history?

    :param page:
    :return:
    """
    j = request.get_json(force=True, silent=True)
    also_description = True if "description" in j else False
    session = DBSession()
    d = session.query(Diagram).filter_by(page=page).first()
    if not d:
        d = Diagram()
        d.page = page
        session.add(d)

    if also_description:
        d.description = j["description"]
        if "content" in j:
            d.content = j["content"]
    else:
        d.content = j
    session.commit()
    r = Response(json.dumps({"id": d.id, "page": d.page, "content": d.content}, default=json_serial,
                            sort_keys=True, indent=JSON_INDENT,
                            ensure_ascii=ENSURE_ASCII, separators=(',', ': ')),
                 mimetype="text/json")
    session.close()
    return r


if __name__ == '__main__':
    import os
    # "cannot connect to X server" error when remote debugging?
    # Execute "Xvfb :99 -ac -noreset" in the remote server and uncomment the following line
    # os.environ["DISPLAY"] = ":99"
    app.run(host='0.0.0.0', debug=True, threaded=True)  # Default port, 5000
