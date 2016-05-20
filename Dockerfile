FROM ubuntu:14.04
MAINTAINER Herval Freire <hervalfreire@gmail.com>

# General dependencies, lots of them
RUN apt-get update && apt-get install -y git
RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev libatlas-dev libzmq3-dev libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler bc libopenblas-dev


# Python + pip
RUN apt-get install -y python python-dev python-pip python-numpy python-scipy


# Caffe
RUN git clone https://github.com/BVLC/caffe.git /caffe
WORKDIR /caffe
RUN cp Makefile.config.example Makefile.config
RUN easy_install --upgrade pip

# Enable CPU-only + openblas (faster than atlas)
RUN sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
RUN sed -i 's/BLAS := atlas/BLAS := open/g' Makefile.config

# Caffe's Python dependencies...
RUN pip install -r python/requirements.txt
RUN make all
RUN make pycaffe
ENV PYTHONPATH=/caffe/python

# Download model
RUN scripts/download_model_binary.py models/bvlc_googlenet


VOLUME ["/data"]


WORKDIR /
ADD deepdream.py /deepdream.py

CMD ["python", "-u", "deepdream.py"]