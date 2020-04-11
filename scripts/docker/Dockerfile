FROM axom/compilers:clang-4
MAINTAINER Jamie Bramwell bramwell1@llnl.gov

USER root

RUN sudo apt-get update -y
RUN sudo apt-get install curl -fy

RUN sudo apt-get -qq install -y --no-install-recommends gfortran-7 && sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-7 100
 
#(For Clang Builds - Need to restore stdc++ lib)
RUN sudo apt-get install libstdc++-7-dev
RUN useradd -ms /bin/bash serac

WORKDIR "/home/serac"
USER serac

RUN git clone --recursive https://github.com/LLNL/serac.git

WORKDIR "/home/serac/serac"

RUN git checkout feature/bramwell/docker

RUN ./scripts/uberenv/uberenv.py --spack-config-dir=./scripts/uberenv/spack_configs/docker --spec=%clang@4.0.0^mpich@3.0.4 --prefix=/home/serac/serac_tpls -k

WORKDIR "/home/serac"

RUN rm -rf serac
