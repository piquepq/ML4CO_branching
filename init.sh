
# install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh || true
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $HOME/anaconda3 || true
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# remove any previous environment
conda env remove -n ml4co

# create the environment from the dependency file
conda env create -n ml4co -f environment.yml

conda activate ml4co

# install package
pip install -e .

