#!/bin/bash

set -e

if [[ "$1" == "" ]]; then
    echo "Usage: $0 <arg>"
    echo "- updatesys: Upgrade system"
    exit 1
fi

if [[ "$1" == "updatesys" ]]; then
    echo "Upgrading system..."
    apt update && apt upgrade -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to upgrade system"
        exit 1
    fi
    echo "System upgraded successfully. Rebooting..."
    reboot
fi

if [[ "$1" == "instantngp" ]]; then
    echo "Installing InstantNGP..."
    apt update
    apt install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev \
                libglew-dev libomp-dev libxinerama-dev libxcursor-dev cmake \
                libjsoncpp-dev libvulkan-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
                g++-12 -y
    
    # Set g++-12 as default compiler
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
    update-alternatives --set gcc /usr/bin/gcc-12
    update-alternatives --set g++ /usr/bin/g++-12
    
    # Install fmt library
    rm -rf fmt
    git clone https://github.com/fmtlib/fmt.git
    cd fmt
    mkdir build && cd build
    cmake .. -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CXX_STANDARD=17
    make -j$(nproc)
    make install
    cd ../..
    
    if [ ! -d "instant-ngp" ]; then
        git clone --recursive https://github.com/nvlabs/instant-ngp --depth=1
    fi
    apt install python3-venv -y
    cd instant-ngp
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-toolkit
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    
    # Configure and build
    export CXX=g++-12
    export CC=gcc-12
    rm -rf build
    cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=86 -DFMT_TEST=OFF -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DFMT_DIR=/usr/local/lib/cmake/fmt
    cmake --build build --config RelWithDebInfo -j$(nproc)
    echo "Done. Run 'source .venv/bin/activate' to activate the virtual environment."
fi

if [[ "$1" == "colmap" ]]; then
    echo "Installing COLMAP..."
    if [ ! -d "colmap" ]; then
        git clone https://github.com/colmap/colmap --depth=1
    fi
    apt update
    apt install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev -y
    cd colmap
    rm -rf build
    mkdir build
    cd build
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    ninja -j$(nproc)
    ninja install
    echo "Done. Run 'colmap' to start COLMAP."
fi

if [[ "$1" == "docker" ]]; then
    echo "Installing Docker..."
    apt-get install ca-certificates curl -y
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt update
    apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
    echo "Done. Run 'docker --version' to check the installation."
fi