# Tag from 19th of August 2024
FROM quay.io/jupyter/minimal-notebook:2024-08-19
    #ubuntu-22.04

FROM quay.io/jupyter/base-notebook:2024-08-19
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
RUN pip install jupyterlab_sos
RUN conda create -y --name versaille python=3.8
RUN conda run -n versaille pip install numpy==1.22.4 &&\
    conda run -n versaille pip install torch==2.4 --index-url https://download.pytorch.org/whl/cpu &&\
    conda install -n versaille -y onnx==1.12 onnxruntime==1.18.0 &&\
    conda run -n versaille pip install stable-baselines3>=2.3 gymnasium    &&\
    conda install -n versaille -y -c anaconda ipykernel &&\
    conda run -n versaille pip install moviepy pygame
USER root

RUN mkdir -p /data && chmod 777 /data && mkdir -p /software && chmod 777 /software

# Julia dependencies
# install Julia packages in /opt/julia instead of ${HOME}
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia \
    JULIA_CPU_TARGET=generic

# Setup Julia
COPY ./setup_julia.py /opt/setup-scripts/setup_julia.py
RUN /opt/setup-scripts/setup_julia.py

USER ${NB_UID}
# Setup IJulia kernel & other packages
RUN /opt/setup-scripts/setup-julia-packages.bash

ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia \
    JULIA_CPU_TARGET=generic

# Download & Install NCubeV
ENV NCUBEV_VERSION=6823867
#0c3c65b
RUN cd /software &&\
    git clone https://github.com/samysweb/NCubeV &&\
    cd NCubeV &&\
    git checkout $NCUBEV_VERSION

ENV JULIA_NUM_PRECOMPILE_TASKS=1 \
    JULIA_CPU_THREADS=1

RUN cd /software/NCubeV/ &&\
    ./build.sh $(which julia)

USER root

ENV JULIA_PROJECT=/software/NCubeV
COPY ./NCubeV /usr/local/bin/NCubeV
RUN chmod 777 /usr/local/bin/NCubeV

USER ${NB_UID}
# Install Julia Versaille Kernel and trigger precompilation as explicitly as possible
COPY setup.jl /software/NCubeV/versaille-setup.jl
RUN julia --project=/software/NCubeV /software/NCubeV/versaille-setup.jl
# Install Python Versaille Kernel
RUN conda run -n versaille python -m ipykernel install --user --name=Python_Versaille

RUN pip install jupyterlab_rise
RUN pip install jupyterlab_sos

WORKDIR /data
COPY ./contents/ /data/
USER root
RUN chmod -R 777 /data
USER ${NB_UID}
VOLUME /data
CMD ["start-notebook.py", "--NotebookApp.token='versaille'"]