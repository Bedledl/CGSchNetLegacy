FROM docker.io/graphcore/pytorch-geometric:3.3.0

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata # this has t be done to prevent interactive tzdata dialog: https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
RUN apt-get update && apt-get install git wget -y
RUN apt-get update && apt-get install vim -y

COPY requirements.txt /opt/requirements.txt

RUN pip3 install -r /opt/requirements.txt

CMD source /opt/poplar/enable.sh && source /opt/popart/enable.sh

