FROM python:3.11

RUN apt-get update && apt-get install -y \
    gcc \
    gdb \
    libhdf5-serial-dev
