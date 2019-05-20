#FROM python:3.7.3-alpine
FROM python:3.7.3-slim-stretch

# The tool to run both "gunicorn" and "Celery" is "supervisor". An example in:
# https://github.com/pm990320/docker-flask-celery
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
# LOCAL SERVER: (remember to start a REDIS instance) the image would be:
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

# ALPINE
#RUN echo "http://dl-cdn.alpinelinux.org/alpine/latest-stable/main" > /etc/apk/repositories
#RUN echo "http://dl-cdn.alpinelinux.org/alpine/latest-stable/community" >> /etc/apk/repositories
##redis gfortran
#RUN apk update
#RUN apk add --no-cache --virtual=.build-dependencies \
#    # lxml
#    libxml2-dev libxslt-dev \
#    lapack \
##    openblas-dev \
#    openblas musl-dev \
##    libpng-dev \
#    git \
##    freetype freetype-dev \
##    build-base \
#    libstdc++ \
#    libffi-dev \
#    postgresql-dev


# NORMAL
RUN apt-get update && \
    apt-get -y install \
    liblapack3  \
    libblas3  \
    gcc \
    git \
    curl \
    libpq-dev \
	libcurl4-openssl-dev \
	libssl-dev \
	mime-support \
	&& apt-get clean


# COMMON

RUN pip install --no-cache-dir git+https://github.com/Supervisor/supervisor

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir gunicorn

ENV MAGIC_NIS_SERVICE_CONFIG_FILE=""

# Generate "requirements.txt" with "pipreqs --force ."
COPY requirements.txt /app
RUN pip3 install --no-cache-dir -r requirements.txt

COPY supervisord.conf /etc/supervisord.conf

COPY backend /app/backend
COPY frontend /app/frontend
RUN mkdir -p /srv

EXPOSE 80
VOLUME /srv

# needs to be set else Celery gives an error (because docker runs commands inside container as root)
ENV C_FORCE_ROOT=1

# run supervisord
CMD ["/usr/local/bin/gunicorn", "--workers=3", "--log-level=debug", "--timeout=2000", "--bind", "0.0.0.0:80", "backend.restful_service.service_main:app"]
#CMD ["supervisord", "-c", "/etc/supervisord.conf"]
