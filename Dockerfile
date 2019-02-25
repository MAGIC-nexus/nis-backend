FROM grahamdumpleton/mod-wsgi-docker:python-3.6

# The image used, "grahamdumpleton/mod-wsgi-docker:python-3.6" is not in "Docker Hub"
# It has to be built from the original source code of 3.5 image (source code in Github), with the following versions
# in the corresponding Dockerfile section:
#
# ENV PYTHON_VERSION=3.6.5 \
#    NGHTTP2_VERSION=1.32.0 \
#    APR_VERSION=1.6.3 \
#    APR_UTIL_VERSION=1.6.1 \
#    APACHE_VERSION=2.4.33 \
#    MOD_WSGI_VERSION=4.6.4 \
#    NSS_WRAPPER_VERSION=1.1.3 \
#    TINI_VERSION=0.18.0
#
# Build the image using the modified Dockerfile, with the name "grahamdumpleton/mod-wsgi-docker:python-3.6"

# TODO
# Rewrite to contain three things:
# * NIS application. Run using "gunicorn"
# * R modules
# * Celery
#
# The tool to run both "gunicorn" and "Celery" is "supervisor". An example in:
# https://github.com/pm990320/docker-flask-celery
#
#
# The present container is for MOD_WSGI (Apache2) <<<<<<<<<<<<<<<<<<
#
#
# Build:
# docker build -t magic-nis .
#
# Usage example:
# docker create --name nis-local -p 8080:80
#               -e MAGIC_NIS_SERVICE_CONFIG_FILE="nis_docker_naples.conf" magic-nis:latest
#
# docker start nis-local && docker logs nis-local -f
#
#
# For local tests (remember to start a REDIS instance) the image would be:
#
# docker create --name nis-local -p 8080:80
#               -v /home/rnebot/DATOS/docker/nis-local:/srv
#               -e MAGIC_NIS_SERVICE_CONFIG_FILE="nis_docker_local_sqlite.conf" magic-nis:latest
#
# NOTE: in the example, the host directory (/home/rnebot/DATOS/docker/nis-local) must have RWX permissions
#       for all users: chmod rwx+o ...
#       If not, it may not be possible to create
#
# PRODUCTION SERVER:
#
# docker create --name nis-local --network=magic-net -l magic-postgis -l magic-redis -v /srv/docker/magic/data/nis:/srv
#   -e VIRTUAL_HOST=one.nis.magic-nexus.eu -e VIRTUAL_PORT=80 -e LETSENCRYPT_HOST=one.nis.magic-nexus.eu
#   -e LETSENCRYPT_EMAIL=rnebot@itccanarias.org -e MAGIC_NIS_SERVICE_CONFIG_FILE="nis_docker_naples.conf"
#   -e MOD_WSGI_REQUEST_TIMEOUT=1500 -e MOD_WSGI_SOCKET_TIMEOUT=1500
##   -e MOD_WSGI_CONNECT_TIMEOUT=1500 -e MOD_WSGI_INACTIVITY_TIMEOUT=1500
#   magic-nis:latest
#
#

ENV MAGIC_NIS_SERVICE_CONFIG_FILE=""

ENV DEBIAN_FRONTEND noninteractive

RUN locale-gen es_ES.UTF-8
ENV LANG es_ES.UTF-8
ENV LANGUAGE=es_ES:es
ENV LC_ALL es_ES.UTF-8

RUN apt-get update && \
    apt-get -y install \
	python3-pip \
        liblapack3  \
        libblas3  \
	python3-scipy

#RUN apt-get -y install wget && \
#    TEMP_DEB="$(mktemp)" && \
#    wget -O "$TEMP_DEB" 'http://security-cdn.debian.org/pool/updates/main/c/curl/curl_7.38.0-4+deb8u14_amd64.deb' && \
#    dpkg -i "$TEMP_DEB" && \
#    rm -f "$TEMP_DEB"

#RUN apt-get -y install libcurl3-openssl-dev

# Generate "requirements.txt" with "pipreqs --force ."
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY backend /app/backend
COPY frontend /app/frontend
RUN mkdir -p /srv

EXPOSE 80
VOLUME /srv

ENTRYPOINT [ "mod_wsgi-docker-start" ]
# (this) Dockerfile -> "nis_docker.wsgi" -> "service_main.py"
CMD [ "/app/backend/restful_service/mod_wsgi/nis_docker.wsgi" ]
