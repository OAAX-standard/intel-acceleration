ARG DEBIAN_FRONTEND=noninteractive
FROM python:3.8.16 as base


RUN echo "## Update and install packages"

RUN echo "### Support packages:"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update -qq -y --fix-missing \
  && apt-get install -qq -y gcc g++ curl make xxd wget apt-utils software-properties-common git autoconf \
    automake libtool python3 unzip cmake build-essential protobuf-compiler libprotoc-dev > /dev/null


WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY conversion_toolchain /app/conversion_toolchain
COPY scripts/ /app/scripts

RUN echo "### Installing Python dependencies:"
RUN pip install -r requirements.txt

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]