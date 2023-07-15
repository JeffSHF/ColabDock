# ColabDock
### Inverting AlphaFold2 structure prediction model for protein-protein docking with experimental restraints

ColabDock is a docking framework developed based on [ColabDesign](https://github.com/sokrypton/ColabDesign.git). It is able to incorporate experimental restraints to generate accurate protein complex structure. For details, please refer to the [paper](https://doi.org/10.1101/2023.07.04.547599).

<a href="https://colab.research.google.com/github/JeffSHF/ColabDock/blob/dev/ColabDock.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Note the notebook is still under development.

### Installation
1. Create a python enviroment using conda.
```bash
conda create --name colabdock python=3.8
```
2. Clone the repo
```bash
git clone git@github.com:JeffSHF/ColabDock.git
```
3. Activate environment
```bash
cd ColabDock && conda activate colabdock
```
4. Install jax  
<b>Please refer to [JAX github](https://github.com/google/jax) page to install package corresponding to your CUDA and cudnn version.</b>  
Example:
```bash
# install jaxlib
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl
# install jax
pip install jax==0.3.8
```

4. Install other dependencies
```bash
pip install -r requirements.txt
```

### Usage
Before running the code, please set the variables in the config.py file according to the descriptive information.
```bash
conda activate colabdock && python main.py
```

### Contributors
- Shihao Feng [@JeffSHF](https://github.com/JeffSHF)
- Zhenyu Chen [@Dreams1de](https://github.com/Dreams1de)
- Sergey Ovchinnikov [@sokrypton](https://github.com/sokrypton)
