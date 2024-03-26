# DEAR: Disentangled Environment and Agent Representations for Reinforcement Learning without Reconstruction

by Ameya Pore, Riccardo Muradore and Diego Dall'Alba

This repo was initially forked from the original [DrQ-v2 repo](https://github.com/facebookresearch/drqv2) and [SEAR](https://github.com/sear-rl/sear-rl).

## Instructions

Install mujoco:
```sh
mkdir ~/.mujoco
wget -P ~/.mujoco https://www.roboti.us/file/mjkey.txt
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux ~/.mujoco/mujoco200
rm mujoco200_linux.zip
```

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Export the following variables (It is recommended to put this into your bashrc or zshrc file)
```sh
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64
export PATH="/usr/local/cuda/bin:$PATH" 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

Install dependencies:
```sh
# Clone this repo
git clone git@github.com:sear-rl/sear-rl.git

# Create a conda environment with all of the dependencies except for metaworld
cd dear
conda env create -f conda_env.yml
conda activate dear


# Install this repo so that imports will work properly
cd ../DEAR
pip install -e .

# Download the DAVIS dataset if you plan on using the distracting-control suite.
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-480p.zip
```
