FROM axom/compilers:gcc-8
MAINTAINER Jamie Bramwell bramwell1@llnl.gov

RUN sudo apt-get update -y
RUN sudo apt-get install curl -fy

RUN sudo apt-get -qq install -y --no-install-recommends gfortran-7 && sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-7 100

RUN sudo useradd -ms /bin/bash serac

WORKDIR "/home/serac"
USER serac

RUN git clone --recursive https://github.com/LLNL/serac.git

WORKDIR "/home/serac/serac"

RUN ./scripts/uberenv/uberenv.py --spack-config-dir=./scripts/uberenv/spack_configs/docker/ --spec=%gcc@8.1.0 --prefix=/home/serac/serac_tpls -k

WORKDIR "/home/serac"

RUN rm -rf serac
