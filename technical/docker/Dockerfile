# Tag from 19th of August 2024
FROM quay.io/jupyter/minimal-notebook:ubuntu-22.04

FROM quay.io/jupyter/base-notebook:ubuntu-22.04
USER root
COPY --from=0 /opt/setup-scripts/ /opt/setup-scripts
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # Common useful utilities
    curl \
    git \
    nano-tiny \
    tzdata \
    unzip \
    build-essential &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Installing python packages
USER ${NB_UID}
RUN conda create -y --name versaille python=3.8
RUN conda run -n versaille pip install torch==2.4 --index-url https://download.pytorch.org/whl/cpu &&\
    conda install -n versaille -y onnx==1.9 onnxruntime==1.18.0 &&\
    conda run -n versaille pip install stable-baselines3>=2.3 gymnasium    &&\
    conda install -n versaille -y -c anaconda ipykernel
RUN conda run -n versaille python -m ipykernel install --user --name=versaille
USER root

RUN mkdir -p /data && chmod 777 /data && mkdir -p /software && chmod 777 /software

# Julia dependencies
# install Julia packages in /opt/julia instead of ${HOME}
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia

# Setup Julia
RUN /opt/setup-scripts/setup_julia.py

USER ${NB_UID}
# Setup IJulia kernel & other packages
RUN /opt/setup-scripts/setup-julia-packages.bash

# Download & Install NCubeV
ENV NCUBEV_VERSION=8aa16ef
RUN cd /software &&\
    git clone https://github.com/samysweb/NCubeV &&\
    cd NCubeV &&\
    git checkout $NCUBEV_VERSION
RUN cd /software/NCubeV/ &&\
    ./build.sh $(which julia)

USER root
RUN ln -s /software/NCubeV/deps/NCubeV/bin/NCubeV /usr/local/bin/NCubeV
#COPY ./julia /software/bin/julia
#RUN mv /usr/local/bin/julia /usr/local/bin/julia-orig && chmod 777 /software/bin/julia && ln -s /software/bin/julia /usr/local/bin/julia
#RUN /software/NCubeV/deps/NCubeV/bin/julia --project=/software/NCubeV/ -E 'using Pkg;Pkg.add(["HDF5","IJulia","Pluto"]);Pkg.precompile()'
#RUN mv /opt/julia-1.10.4/lib/ /opt/julia-1.10.4/lib-old
#RUN ln -s /software/NCubeV/deps/NCubeV/lib /opt/julia-1.10.4/lib

ENV JULIA_PROJECT=/software/NCubeV

USER ${NB_UID}


WORKDIR /data