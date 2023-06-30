FROM nvcr.io/nvidia/pytorch:22.03-py3

RUN apt-get update && apt-get install -y curl wget nano iproute2 iputils-ping
RUN pip install --upgrade pip
RUN apt install git -y
RUN update-alternatives --install "/usr/bin/python" "python" "$(which python3)" 1

WORKDIR /workspace
COPY . .

RUN pip install transformers

EXPOSE 8888
