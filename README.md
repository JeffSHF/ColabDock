# ColabDock
<b>Inverting AlphaFold2 structure prediction model for protein-protein docking with experimental restraints</b>

![ColabDock](https://github.com/JeffSHF/ColabDock/assets/88184243/62b65508-6bbf-46f5-a4c0-72206b5e09fe)
ColabDock is a docking framework developed based on [ColabDesign](https://github.com/sokrypton/ColabDesign.git). It is able to incorporate experimental restraints to generate accurate protein complex structure. For details, please refer to the [paper](https://doi.org/10.1101/2023.07.04.547599).

Note running ColabDock locally requires GPU. If you do not have one, we suggest using the Colab version.
<a href="https://colab.research.google.com/github/JeffSHF/ColabDock/blob/dev/ColabDock.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<b>Table of content</b>
- [ColabDock](#colabdock)
    - [Installation on Linux](#installation-on-linux)
    - [Usage](#usage)
    - [Restraints sampling](#restraints-sampling)
    - [Restraints example](#restraints-example)
      - [a. 1v1 restraint](#a-1v1-restraint)
      - [b. 1vN restraint](#b-1vn-restraint)
      - [c. MvN restraint](#c-mvn-restraint)
      - [d. Repulsive restraint](#d-repulsive-restraint)
    - [Restraints used in the paper](#restraints-used-in-the-paper)
    - [Links](#links)
    - [Contributors](#contributors)


### Installation on Linux
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
Example (which has been tested locally):
```bash
# install jaxlib
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl
# install jax
pip install jax==0.3.8
```

5. Install other dependencies
```bash
pip install -r requirements.txt
```

6. Download AlphaFold2 parameters
```bash
mkdir params
cd params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xvf alphafold_params_2022-12-06.tar
```
The installation takes ~10 min.

### Usage
Before running the code, please set the variables in the config.py file according to the descriptive information in it.
The default config.py contains the setting of protein 4INS4.
```bash
conda activate colabdock && python main.py
```
The running time depends on the size of the protein, the round and step numbers. For the default setting, it should take ~10 min.

After the running, the outputs directory (default is `results`) will contain three folders, i.e., gen, pred, and docked. Gen and pred contain the pdb files of all the docking structures derived from the generation stage and the prediction stage. The docked folder contains the top5 predicted docking structures.


### Restraints sampling
If you want to test ColabDock using a complex with known structure, you can generate 1v1, 1vN, or MvN restraints using the extract_rest.py script. For example, 4INS4 contains 4 protein chains, i.e., A,B,C,D. If you want to sample some 1v1 restraints between chain A and chain D, then run the following command and the program will print the sampled restraints.
```python
python extract_rest.py ./protein/4INS4/PDB/native.pdb A,B,C,D A,D 1v1
```
You can also use `python extract_rest.py -h` to get more informations about the input parameters.


### Restraints example
Suppose the complex you want to dock contains two chains, i.e., A and B, and each chain contains 10 amino acids. The "chains" variable in the config.py file is set to `'A,B'`.
#### a. 1v1 restraint
If you want the 4th residue in chain A close to the 5th residue in chain B in the docking structure, the 1v1 restraint (`rest_1v1` variable in the config.py file) should be set to `[4, 15]`.  
#### b. 1vN restraint
If you want the 4th residue in chain A close to one of residues from 3rd to 7th in chain B in the docking structure, the 1vN restraint (`rest_1vN` variable in the config.py file) should be set to `[4, range(13, 18)]`.  
#### c. MvN restraint
If you have two 1vN restraints, i.e., `[4, range(13, 18)]` and `[6, range(13, 18)]`, and you want at least one of them satisfied in the docking structure, the MvN restraint (`rest_MvN` variable in the config.py file) should be set to `[[4, range(13, 18)], [6, range(13, 18)], 1]`.  
#### d. Repulsive restraint
If you want the 6th residue in chain A far away from the 8th residue in chain B in the docking structure, the repulsive restraint (`rest_rep` variable in the config.py file) should be set to `[6, 18]`.  

### Restraints used in the paper
In ColabDock paper, we sampled restraints for proteins collected from [protein docking benchmark 5.5](https://doi.org/10.1016/j.jmb.2015.07.016). The sampled restraints can be downloaded from [OSF](https://doi.org/10.17605/OSF.IO/N6R48).

### Links
A blog in Chinese ([link](https://mp.weixin.qq.com/s/7-GE5Ueyq-7IpaezWUTyZA))

### Contributors
- Shihao Feng [@JeffSHF](https://github.com/JeffSHF)
- Zhenyu Chen [@Dreams1de](https://github.com/Dreams1de)
- Sergey Ovchinnikov [@sokrypton](https://github.com/sokrypton)
