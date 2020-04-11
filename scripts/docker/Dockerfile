FROM ubuntu:18.04
MAINTAINER Jamie Bramwell bramwell1@llnl.gov

RUN apt-get update -y
RUN apt-get install libtool git curl gcc gfortran python cmake -fy

RUN useradd -ms /bin/bash serac

WORKDIR "/home/serac"
USER serac

RUN git clone --recursive https://github.com/LLNL/serac.git

WORKDIR "/home/serac/serac"

#RUN ./scripts/uberenv/uberenv.py --spack-config-dir=./scripts/uberenv/spack_configs/linux-ubuntu18.04-broadwell/ --spec=%gcc --prefix=/home/serac/serac_tpls -k

#WORKDIR "/home/serac"

#RUN rm -rf serac
