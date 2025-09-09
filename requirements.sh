#Create a virtual environment 
conda create -n plmalign python=3.9
conda activate plmalign

#conda environment
conda install -c conda-forge biopython
conda install -c conda-forge tqdm
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
conda install -c conda-forge scikit-learn
conda install ipykernel --update-deps --force-reinstall
conda install dask -c conda-forge
conda install matplotlib -c conda-forge 

pip install torch torchvision
# Or other pytorch versions depending on your local environment
pip install numpy==1.26
pip install scipy
pip install numba
pip install transformers
pip install sentencepiece
pip install fair-esm