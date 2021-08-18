conda create -y -n pl python=3.8 pip
conda activate pl
conda install -y pytorch=1.8 torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install pytorch-lightning==1.2
conda install -y pandas scikit-image scikit-learn 
conda install -y -c conda-forge gdcm

pip install albumentations kaggle iterative-stratification omegaconf pydicom timm==0.4.5 transformers
