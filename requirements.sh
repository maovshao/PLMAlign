#Create a virtual environment 
conda create -n plmalign python=3.9
conda activate plmalign

#conda environment
conda install -c conda-forge biopython
conda install -c conda-forge tqdm
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
#(Option for CPU only) pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
conda install -c conda-forge scikit-learn
conda install ipykernel --update-deps --force-reinstall
conda install dask -c conda-forge
conda install matplotlib -c conda-forge 
conda install seaborn -c conda-forge

pip install scipy
pip install numba
pip install transformers
pip install sentencepiece
pip install fair-esm