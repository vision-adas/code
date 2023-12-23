FROM nvcr.io/nvidia/tensorrt:22.03-py3
#FROM adas:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y debconf-utils
RUN apt install -y keyboard-configuration

RUN apt -y install gdb
RUN apt install -y git cmake g++ wget
RUN apt install -y libeigen3-dev libgphoto2-dev libpcl-dev libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip wget rsync libgtest-dev

RUN wget https://github.com/opencv/opencv/archive/refs/tags/4.5.5.zip
RUN unzip 4.5.5.zip
RUN cd opencv-4.5.5\
&& mkdir build\
&& cd build\
&& cmake -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLE=off -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON ..\
&& make -j 8\
&& make install

RUN pip install opencv-python
