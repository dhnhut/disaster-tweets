# Vultr Cloud Setup Script
# Ubuntu 26.04 LTS x64

# !/bin/bash
sudo apt update
sudo apt install git
sudo apt install git-lfs

# # # # Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# `-b` run install in batch mode (without manual intervention), it is expected the license terms (if any) are agreed upon
bash ~/Miniconda3-latest-Linux-x86_64.sh -b
# # Recommended: initialize conda for bash
# # vultrcomputer
# sudo /root/miniconda3/bin/conda init bash
# thudercomputer
./miniconda3/bin/conda init bash
source ~/.bashrc

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda install -n base ipykernel --update-deps --force-reinstall -y

conda config --add channels conda-forge
pip install datasketch transformers torch tqdm datasets scikit-learn scikit-learn python-dotenv matplotlib peft trl bitsandbytes evaluate

git clone https://github.com/dhnhut/disaster-tweets
cd disaster-tweets
git lfs install
git lfs pull
git config --global user.name "Nhut Hoang Duong"
git config --global user.email dhnhut@gmail.com

# pip install -r disaster-tweets/requirements.txt

# # https://arcolinux.com/how-to-increase-the-size-of-your-swapfile/
# sudo dd if=/dev/zero of=/swapfile bs=1G count=512
# sudo mkswap /swapfile
# sudo swapon /swapfile
# # # Check swap size
# # grep SwapTotal /proc/meminfo