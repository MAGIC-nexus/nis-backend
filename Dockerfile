FROM grahamdumpleton/mod-wsgi-docker:python-3.5

# TODO
# Rewrite to contain three things:
# * NIS application. Run using "gunicorn"
# * R modules
# * Celery
# The tool is "supervisor". An example in:
# https://github.com/pm990320/docker-flask-celery
#
#
# This one container is for MOD_WSGI (Apache2) <<<<<<<<<<<<<<<<<<
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
# Production server:
# docker create --name nis-local -l magic-postgis -l magic-redis -v /srv/docker/magic/data/nis:/srv
#   -e VIRTUAL_HOST=one.nis.magic-nexus-eu -e VIRTUAL_PORT=80 -e LETSENCRYPT_HOST=one.nis.magic-nexus-eu
#   -e LETSENCRYPT_EMAIL=rnebot@itccanarias.org -e MAGIC_NIS_SERVICE_CONFIG_FILE="nis_docker_naples.conf"
#   magic-nis:latest
#
#
# NOTE: in this last example, the host directory (/home/rnebot/DATOS/docker/nis-local) must have RWX permissions
#       for all users: chmod rwx+o ...
#       If not, it may not be possible to create

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
