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
Before running the code, please set the variables in the config.py file according to the descriptive information in it.
```bash
conda activate colabdock && python main.py
```

### Restraints example
Suppose the complex you want to dock contains two chains, i.e., A and B, and each chain contains 10 amino acids. The "chains" variable in the config.py file is set to `'A,B'`.
#### a. 1v1 restraint
If you want the 4th residue in chain A is close to the 5th residue in chain B in the docking structure, the 1v1 restraint (`rest_1v1` variable in the config.py file) should be set to `[4, 15]`.  
#### b. 1vN restraint
If you want the 4th residue in chain A is close to one of residues from 3rd to 7th in chain B in the docking structure, the 1vN restraint (`rest_1vN` variable in the config.py file) should be set to `[4, range(13, 18)]`.  
#### c. MvN restraint
If you have two 1vN restraints, i.e., `[4, range(13, 18)]` and `[6, range(13, 18)]`, and you want at least one of them satisfied in the docking structure, the MvN restraint (`rest_MvN` variable in the config.py file) should be set to `[[4, range(13, 18)], [6, range(13, 18)], 1]`.  
#### d. Repulsive restraint
If you want the 6th residue in chain A is far away from the 8th residue in chain B in the docking structure, the repulsive restraint (`rest_rep` variable in the config.py file) should be set to `[6, 18]`.  


### Link
A blog in Chinese ([link](https://mp.weixin.qq.com/s/7-GE5Ueyq-7IpaezWUTyZA))

### Contributors
- Shihao Feng [@JeffSHF](https://github.com/JeffSHF)
- Zhenyu Chen [@Dreams1de](https://github.com/Dreams1de)
- Sergey Ovchinnikov [@sokrypton](https://github.com/sokrypton)
