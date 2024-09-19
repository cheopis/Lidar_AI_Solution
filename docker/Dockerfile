ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7" 

# The cudnn version for this image is 8.2
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Avoid Public GPG key error
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install necessary libraries
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libprotobuf-dev sudo lsof libncurses5-dev libncursesw5-dev libssl-dev \
    && apt-get install -y gcc g++ python3 git wget python3-pip nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN wget https://cmake.org/files/v3.18/cmake-3.18.5.tar.gz \
    && tar zxvf cmake-3.18.5.tar.gz && rm -f cmake-3.18.5.tar.gz && cd cmake-3.18.5 \
    && ./bootstrap && make -j8 && make install
RUN rm -rf cmake-3.18.5

RUN pip3 install onnx opencv-python

ARG TENSORRT_VERSION=8.2.2.1
ARG CUDA_USER_VERSION=10.2
ARG CUDNN_USER_VERSION=8.2
ARG OPERATING_SYSTEM=Linux

# Install TensorRT
# RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
COPY ../TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.cudnn${CUDNN_USER_VERSION}.tar.gz /opt
RUN cd /opt && \
    tar -xzf TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.cudnn${CUDNN_USER_VERSION}.tar.gz && \
    rm TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.cudnn${CUDNN_USER_VERSION}.tar.gz && \
    export PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 | tr -d .) && \
    python3 -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl && \
    #python3 -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt_lean-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl && \
    #python3 -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt_dispatch-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl && \
    python3 -m pip install TensorRT-${TENSORRT_VERSION}/onnx_graphsurgeon/onnx_graphsurgeon-*-py2.py3-none-any.whl

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-${TENSORRT_VERSION}/lib
ENV PATH=$PATH:/opt/TensorRT-${TENSORRT_VERSION}/bin


ADD . /workspace

# ==================
# Install OPencv
# ==================

ARG OPENCV_VERSION="4.1.1"

# Install build dependencies
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential binutils \
        ca-certificates cmake cmake-qt-gui curl \
        dbus-x11 \
        ffmpeg \
        gdb gcc g++ gfortran git \
        tar \
        lsb-release \
        procps \
        manpages-dev \
        unzip \
        zip \
        wget \
        xauth \
        swig \
        python3-pip python3-dev python3-numpy python3-distutils \
        python3-setuptools python3-pyqt5 python3-opencv \
        libboost-python-dev libboost-thread-dev libatlas-base-dev libavcodec-dev \
        libavformat-dev libavutil-dev libcanberra-gtk3-module libeigen3-dev \
        libglew-dev libgl1-mesa-dev libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        libgtk-3-dev libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev liblapack-dev \
        liblapacke-dev libopenblas-dev libopencv-dev libpng-dev libpostproc-dev \
        libpq-dev libsm6 libswscale-dev libtbb-dev libtbb2 libtesseract-dev \
        libtiff-dev libtiff5-dev libv4l-dev libx11-dev libxext6 libxine2-dev \
        libxrender-dev libxvidcore-dev libx264-dev libgtkglext1 libgtkglext1-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev libopenexr-dev \
        openexr \
        pkg-config \
        qv4l2 \
        v4l-utils \
        zlib1g-dev \
        locales \
        && locale-gen en_US.UTF-8 \
        && LC_ALL=en_US.UTF-8 \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# libvtk9-dev libdc1394-dev

WORKDIR /opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv-${OPENCV_VERSION} opencv \
    && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

RUN mkdir /opencv/opencv/build
WORKDIR /opencv/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -DCMAKE_INSTALL_PREFIX=/usr/local \
 -DENABLE_CXX11=ON                \
 -DBUILD_PERF_TESTS=OFF           \
 -DWITH_XINE=ON                   \
 -DBUILD_TESTS=OFF                \
 -DENABLE_PRECOMPILED_HEADERS=OFF \
 -DCMAKE_SKIP_RPATH=ON            \
 -DBUILD_WITH_DEBUG_INFO=OFF ..   \
    && make -j$(nproc) && make install && ldconfig

RUN git clone https://github.com/nothings/stb.git \
    && sudo cp stb/stb_image.h /usr/local/include/ \
    && sudo cp stb/stb_image_write.h /usr/local/include/

WORKDIR /workspace/libraries/spconv 
RUN python setup.py bdist_wheel \
    && cp build/lib.linux-x86_64-3.7/spconv/libspconv.so /usr/local/include \
    && cp build/lib.linux-x86_64-3.7/spconv/libspconv.so /usr/local/lib \

# docker build -t jetson -f ./docker/Dockerfile .
# docker run --gpus all --shm-size=8g -it jetson
# docker exec -ti container_name /bin/bash