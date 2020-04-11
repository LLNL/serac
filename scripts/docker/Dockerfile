FROM axom/compilers:gcc-8
MAINTAINER Jamie Bramwell bramwell1@llnl.gov

RUN sudo apt-get update -y
RUN sudo apt-get install curl ssh pkg-config build-essential gfortran libgfortran3 libgfortran4 libopenblas-dev -fy

RUN sudo useradd -m -s /bin/bash -G sudo serac

WORKDIR "/home/serac"
USER serac

RUN git clone --recursive https://github.com/LLNL/serac.git

WORKDIR "/home/serac/serac"

RUN git checkout feature/bramwell/docker

RUN ./scripts/uberenv/uberenv.py --spack-config-dir=./scripts/uberenv/spack_configs/docker/ --spec=%gcc@8.1.0 --prefix=/home/serac/serac_tpls -k

WORKDIR "/home/serac"

RUN rm -rf serac
