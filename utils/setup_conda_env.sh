ENV_NAME=iner_2025
PROJ_ROOT=/specific/scratches/parallel/evyataroren-2025-12-31/iner_2025/
CUDA_VERSION=124


cd $PROJ_ROOT
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p .miniconda3
.miniconda3/bin/conda init bash
rm Miniconda3-latest-Linux-x86_64.sh
bash
source ~/.bashrc
conda create --name $ENV_NAME python=3.11 -y
conda activate $ENV_NAME


python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support from official PyTorch wheels
python -m pip install torch --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
conda install -c conda-forge cuda-nvcc -y
python -m pip install -r env/requirements.txt -y