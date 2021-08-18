conda create -y -n pl python=3.8 pip
conda activate pl
conda install -y pytorch=1.8 torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install pytorch-lightning==1.2
conda install -y pandas scikit-image scikit-learn 
conda install -y -c conda-forge gdcm

pip install albumentations kaggle iterative-stratification omegaconf pydicom timm==0.4.5 transformers


# Install fairscale to support sharded multi-GPU training
# Might need to run so that CUDA uses updated compiler:
sudo yum install gcc72 gcc72-c++
sudo ln -s /usr/bin/gcc72 /usr/local/cuda/bin/gcc
pip install fairscale