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
conda install datasketch transformers pytorch tqdm datasets scikit-learn python-dotenv matplotlib peft trl bitsandbytes evaluate sentence-transformers gdown -y

# Only install one of them, depending on whether you have GPU support or not
# conda install faiss-cpu -y
conda install faiss-gpu -y

git clone https://github.com/dhnhut/disaster-tweets
cd disaster-tweets
git lfs install
# git lfs pull
git config --global user.name "Nhut Hoang Duong"
git config --global user.email dhnhut@gmail.com

# pip install -r disaster-tweets/requirements.txt

# # https://arcolinux.com/how-to-increase-the-size-of-your-swapfile/
# sudo dd if=/dev/zero of=/swapfile bs=1G count=512
# sudo mkswap /swapfile
# sudo swapon /swapfile
# # # Check swap size
# # grep SwapTotal /proc/meminfo

# df_disaster_informative_knearest_0.75_100.csv
# gdown https://drive.google.com/file/d/1H5d2b2h7ace3uCR3Fkohw-1Wyk5oaf7t/view?usp=drive_link -O ./data/disaster/

# df_weather_radius_0.75.csv
# gdown https://drive.google.com/file/d/1iXC8QT8xLqGl8uQ7Imgyydfe1cRIQyVp/view?usp=drive_link -O ./data/extended/

# df_out_topic_knearest_0.6_top_100.csv
# gdown https://drive.google.com/file/d/1m43_waNzeiCzA1WpjYvMujO0CFsDQ2Yu/view?usp=drive_link -O ./data/extended/