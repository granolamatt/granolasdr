FROM nvidia/cuda:12.6-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libusb-1.0-0-dev \
    libzmq3-dev \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Rust 1.95.0 — pinned for reproducibility (wsserver Cargo.lock pins crates)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.95.0 --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"

# pybind11 required by CMakeLists.txt (Python bindings target)
RUN pip3 install pybind11

# libsddc — RX888 driver (granolamatt/ExtIO_sddc pinned to b3ff960)
RUN git clone https://github.com/granolamatt/ExtIO_sddc /tmp/ExtIO_sddc && \
    cd /tmp/ExtIO_sddc && \
    git checkout b3ff9604667e5534e58b4e0cf7c51f69bb7a8850 && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    ldconfig && \
    rm -rf /tmp/ExtIO_sddc

WORKDIR /workspace
COPY . .

# CMakeLists.txt hardcodes /opt/cuda; Ubuntu CUDA installs to /usr/local/cuda
RUN ln -s /usr/local/cuda /opt/cuda

# Pre-fetch Rust crates into the layer cache before cmake triggers cargo build
RUN cd wsserver && cargo fetch

RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90;120" \
          .. && \
    make -j$(nproc) hf_rx

EXPOSE 8765 5599 5600 5581 5582 5583 5584

CMD ["./build/hf_rx"]
