# syntax=docker/dockerfile:1.3
ARG BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu20.04
FROM ${BASE_IMAGE} as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

ARG USERNAME=user
ARG UID=1000
ARG GID=1000

# ---------------------------
# System packages
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo gnupg2 curl wget ca-certificates \
    build-essential cmake git g++ gdb pkg-config ninja-build \
    python3-dev python3-pip python3-tk \
    libeigen3-dev libopencv-dev libfmt-dev \
    freeglut3-dev \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev \
    mesa-common-dev \
    vulkan-utils mesa-vulkan-drivers \
    libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    zip unzip pigz \
    ffmpeg \
    git-lfs \
  && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Optional: avoid Mesa EGL being selected (keep your original intent)
# Make it non-fatal if paths differ in CUDA12 images.
# ---------------------------
RUN rm -f \
    /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 \
    /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 \
    /usr/share/glvnd/egl_vendor.d/50_mesa.json \
  || true

COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ---------------------------
# Install Eigen 3.4.0 (you previously downloaded it at runtime)
# Put it in /opt so it's always available.
# ---------------------------
ARG EIGEN_VERSION=3.4.0
RUN mkdir -p /opt && cd /opt \
  && wget -q https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.zip \
  && unzip -q eigen-${EIGEN_VERSION}.zip \
  && rm -f eigen-${EIGEN_VERSION}.zip
ENV EIGEN_ROOT=/opt/eigen-${EIGEN_VERSION}

# ---------------------------
# User setup
# ---------------------------
RUN (getent group ${GID} >/dev/null || groupadd -g ${GID} ${USERNAME}) \
  && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} \
  && usermod -aG sudo,video ${USERNAME} \
  && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
  && chmod 440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV HOME="/home/${USERNAME}"
ENV PATH="${PATH}:${HOME}/.local/bin"

# (Optional QoL) PATH in bashrc
RUN echo 'export PATH="$HOME/.local/bin:${PATH}"' >> "${HOME}/.bashrc"

# CUDA envs
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
ENV PATH="${PATH}:${CUDA_HOME}/bin"

# RTX 5090: build CUDA extensions with sm_120 + PTX fallback
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"

# ---------------------------
# Python base tooling
# ---------------------------
RUN python3 -m pip install --upgrade pip wheel setuptools Cython

# ---------------------------
# Pre-torch dependencies (safe to install without torch)
# Pin the ones that are sensitive on Python3.8.
# ---------------------------
RUN python3 -m pip install --upgrade \
    pybind11 \
    cmake-build-extension \
    shapely \
    chama \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-image==0.21.0 \
    triangle \
    rtree \
    fvcore \
    iopath \
    tqdm \
    einops \
    'pyglet<2' \
    gym==0.23.1

# ---------------------------
# Install your custom torch wheel(s)
# You said you put torch*.whl next to Dockerfile.
# This glob can include torch/torchvision/torchaudio if they start with "torch".
# ---------------------------

COPY --chown=${UID}:${GID} torch*.whl /tmp/
RUN python3 -m pip install /tmp/torch*.whl && rm -f /tmp/torch*.whl

# cuDNN 9 is needed for torch to be importable during build (e.g. nvdiffrast)
RUN python3 -m pip install nvidia-cudnn-cu12
ENV LD_LIBRARY_PATH="${HOME}/.local/lib/python3.8/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# libopenblas is needed by torch at import time (used during torchvision build)
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends libopenblas-dev \
  && sudo rm -rf /var/lib/apt/lists/*

# torchvision must be built from source against our custom torch;
# the PyPI wheel was compiled against official torch 2.3.1 and its C++ ops won't load.
RUN python3 -m pip install --no-deps --no-build-isolation \
    git+https://github.com/pytorch/vision.git@v0.18.1

# ---------------------------
# Torch-dependent / heavier packages
# (Keep unpinned unless they're known to break on py3.8)
# ---------------------------
RUN python3 -m pip install --upgrade \
    tensorboard \
    trimesh \
    open3d==0.18.0 \
    pybullet_rendering \
    pyrender \
    opencv-python \
    yourdfpy \
    pysdf \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    transformers==4.40.0 \
    wandb \
    line_profiler \
    cachetools \
    nvtx \
    webdataset \
    graspnetAPI \
    autopep8 \
    pyflakes \
    mypy \
    icecream \
    torchviz \
    gdown \
    moviepy \
    opt_einsum \
    coacd \
    pymeshlab \
    kornia

# Re-pin numpy: some packages above silently downgrade it to 1.20.x,
# but trimesh and torch both need >= 1.21 (for numpy.typing.NDArray).
RUN python3 -m pip install numpy==1.24.4

RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# timm is needed by mvp; install with --no-deps so it doesn't pull PyTorch from PyPI
# (our custom torch 2.3.0a0 is seen as pre-release and pip would replace it)
RUN python3 -m pip install --no-deps timm

ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"

# Git installs: --no-deps prevents pip from resolving torch and overwriting our custom build
RUN python3 -m pip install --no-build-isolation --no-deps \
    git+https://github.com/NVlabs/nvdiffrast.git \
    git+https://github.com/ir413/mvp \
    git+https://github.com/facebookresearch/pytorch3d.git \
    flash-attn
RUN python3 -m pip install --upgrade \
    git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

# ---------------------------
# Runtime helper script
# ---------------------------


ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

CMD ["/bin/bash"]
