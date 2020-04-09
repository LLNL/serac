FROM axom/compilers:clang-4

RUN sudo apt-get update -y
RUN sudo apt-get install curl -fy

RUN sudo apt-get -qq install -y --no-install-recommends gfortran-7 && sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-7 100
 
#(For Clang Builds - Need to restore stdc++ lib)
RUN sudo apt-get install libstdc++-7-dev

RUN git clone --recursive https://github.com/LLNL/serac.git

RUN ./serac/scripts/uberenv/uberenv.py --spack-config-dir=./scripts/uberenv/spack_configs/docker --spec=%clang@4.0.0^mpich@3.0.4 --prefix=/home/jabramwell/serac_tpls -k

RUN rm -rf serac
