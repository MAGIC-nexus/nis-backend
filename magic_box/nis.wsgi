import sys

import os
code_path = '/var/www/swmagic/'
os.environ["MAGIC_SERVICE_CONFIG_FILE"] = code_path + "magic_box/itc.conf"

if code_path not in sys.path:
    sys.path.insert(0, code_path)
print(sys.path)

import magic_box.monitor as monitor
monitor.start(interval=1.0)
monitor.track(os.path.join(os.path.dirname(__file__), 'site.cf'))

from magic_box.service import app as application


#def application(environ, start_response):
#    status = '200 OK'
#    output = 'Hello World!'

#    response_headers = [('Content-type', 'text/plain'),
#                        ('Content-Length', str(len(output)))]
#    start_response(status, response_headers)

#    return [output]
