import os
import argparse
import random
import joblib
import numpy as np
from Bio.PDB.PDBParser import PDBParser

from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.model import model

def gen_1vN_from_1v1(x, y, dm, N_1vN, rest_thres):
    iy_new = [y]
    y_pos = np.where((dm[x, :]>1) & (dm[x, :]<rest_thres))[0]
    y_pos = list(y_pos)
    y_pos.remove(y)
    pos_num = random.choice(range(N_1vN))
    pos_num = min(pos_num, len(y_pos))
    if pos_num > 0:
        y_pos_sel = random.sample(y_pos, pos_num)
        iy_new.extend(y_pos_sel)
    if len(iy_new) < N_1vN:
        y_neg = np.where(dm[x, :]>rest_thres)[0]
        neg_num = N_1vN - len(iy_new)
        y_neg_sel = random.sample(list(y_neg), neg_num)
        iy_new.extend(y_neg_sel)
    return iy_new

def gen_neg_1vN(dm, N_1vN, rest_thres):
    x, _ = np.where(dm > rest_thres)
    ix = random.choice(x)
    iys = np.where(dm[ix, :] > rest_thres)[0]
    iys = random.sample(list(iys), N_1vN)
    return [ix, iys]


if __name__ == '__main__':
    description = 'randomly extract restraints from a given complex structure'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pdb', type=str, help='input complex structure')
    parser.add_argument('chains', type=str, help='docking chains, this parameter should be the same as that in the config.py file')
    parser.add_argument('chains_rest', type=str, help='sample restraints between which two chains')
    parser.add_argument('rest_type', type=str, help='sampled restraints type. Please choose one type from 1v1, 1vN, MvN, and repulsive')
    parser.add_argument('--rest_thres', type=float, default=8.0, help='distance threshold of the restraints. Default 8.0')
    parser.add_argument('--N', type=int, default=5, help='In the sampled 1vN restraint, the distance between a residue and at least one of N residues is below a certain value. Default 5.')
    parser.add_argument('--M', type=int, default=2, help='number of 1vN restraints in the sampled MvN restraint. Default 2.')
    parser.add_argument('--num', type=int, default=1, help='number of sampled restraints. Default 1.')
    parser.add_argument('--save_path', type=str, default=None, help='the file to save the sampled restraints using joblib package. Default None.')

    # parse argument
    args = parser.parse_args()
    file = args.pdb
    chains = args.chains
    chains_rest = args.chains_rest
    rest_type = args.rest_type
    rest_thres = args.rest_thres
    N_1vN = args.N
    M_MvN = args.M
    num = args.num
    save_path = args.save_path

    # check argument
    if not os.path.exists(file):
        raise Exception('The pdb file you provide does not exist!')
    
    if rest_type not in ['1v1', '1vN', 'MvN', 'repulsive']:
        raise Exception('The rest_type argument accepts 1v1, 1vN, MvN, or repulsive.')
    
    if N_1vN <= 1:
        raise Exception('N_1vN should be larger than 1.')
    
    if num < 1:
        raise Exception('At least generate one restraint.')
    
    chains_all_l = [c.strip() for c in chains.split(',')]
    chains_rest_l = [c.strip() for c in chains_rest.split(',')]

    pdb_parser = PDBParser(QUIET=True)
    structures = pdb_parser.get_structure('none', file)
    structure = list(structures.get_models())[0]
    for ichain in chains_all_l:
        if ichain not in structure:
            raise Exception(f'Chain {ichain} is not in the provided pdb file!')
    
    for ichain in chains_rest_l:
        if ichain not in chains_all_l:
            raise Exception(f'Chain {ichain} in chains_rest is not in chains!')
    
    if len(set(chains_rest_l)) != 2 or len(chains_rest_l) != 2:
        raise Exception('Currently, this script only generates restraints between two chains.')

    # cal distance matrix
    pdb = prep_pdb(file, chain=chains, for_alphafold=False)
    x_beta, _ = model.modules.pseudo_beta_fn(aatype=pdb['batch']['aatype'],
                                             all_atom_positions=pdb['batch']["all_atom_positions"],
                                             all_atom_mask=pdb['batch']["all_atom_mask"])
    dm = np.sqrt(np.square(x_beta[:,None] - x_beta[None,:]).sum(-1))

    # sample restraints
    lens = [(pdb["idx"]["chain"] == c).sum() for c in chains_all_l]
    boundaries = [0] + list(np.cumsum(lens))
    ind = chains_all_l.index(chains_rest_l[0])
    a_start, a_stop = boundaries[ind], boundaries[ind+1]
    ind = chains_all_l.index(chains_rest_l[1])
    b_start, b_stop = boundaries[ind], boundaries[ind+1]
    mask = np.zeros_like(dm)
    mask[a_start:a_stop, b_start:b_stop] = 1
    dm *= mask
    
    if rest_type == '1v1':
        x, y = np.where((dm>1) & (dm<rest_thres))
        if len(x) == 0:
            raise Exception('No restraint between the provided two chains!')
        if num > len(x):
            print(f'Cannot sample {num} restraints. There are only {len(x)} restraints between the provided two chains. '
                  f'Consider to generate {len(x)} restraints!')
            num = len(x)

        idx = random.sample(range(len(x)), num)
        x = [x[ind] for ind in idx]
        y = [y[ind] for ind in idx]
        rest = [[x[i]+1, y[i]+1] for i in range(len(x))]
    elif rest_type == 'repulsive':
        x, y = np.where(dm > rest_thres)
        if len(x) == 0:
            raise Exception('No restraint between the provided two chains!')
        if num > len(x):
            print(f'Cannot sample {num} restraints. There are only {len(x)} restraints between the provided two chains. '
                  f'Consider to generate {len(x)} restraints!')
            num = len(x)

        idx = random.sample(range(len(x)), num)
        x = [x[ind] for ind in idx]
        y = [y[ind] for ind in idx]
        rest = [[x[i]+1, y[i]+1] for i in range(len(x))]
    elif rest_type == '1vN':
        x, y = np.where((dm>1) & (dm<rest_thres))
        if len(x) == 0:
            raise Exception('No restraint between the provided two chains!')
        if num > len(x):
            print(f'Cannot sample {num} restraints. There are only {len(x)} restraints between the provided two chains. '
                  f'Consider to generate {len(x)} restraints!')
            num = len(x)

        idx = random.sample(range(len(x)), num)
        x = [x[ind] for ind in idx]
        y = [y[ind] for ind in idx]
        y_new = [gen_1vN_from_1v1(x[i], y[i], dm, N_1vN, rest_thres) for i in range(len(x))]
        rest = [[x[i]+1, list(np.array(y_new[i])+1)] for i in range(len(x))]
    elif rest_type == 'MvN':
        x, y = np.where((dm>1) & (dm<rest_thres))
        if len(x) == 0:
            raise Exception('No restraint between the provided two chains!')
        if num > len(x):
            print(f'Cannot sample {num} restraints. There are only {len(x)} restraints between the provided two chains. '
                  f'Consider to generate {len(x)} restraints!')
            num = len(x)

        rest_1v1 = [[x[i], y[i]] for i in range(len(x))]
        random.shuffle(rest_1v1)
        rest_1v1 = np.array(rest_1v1, dtype=np.int32)
        x, y = list(rest_1v1[:, 0]), list(rest_1v1[:, 1])

        num_1vN = len(x) // num
        posnum_1v1 = [random.choice(range(1, min(num_1vN, M_MvN)+1)) for _ in range(num)]
        rest = []
        for i in range(num):
            iMvN = []
            for j in range(posnum_1v1[i]):
                ix, iy = x[num_1vN*i+j], y[num_1vN*i+j]
                iMvN.append([ix+1, list(np.array(gen_1vN_from_1v1(ix, iy, dm, N_1vN, rest_thres))+1)])
            for j in range(M_MvN-posnum_1v1[i]):
                neg_1vN = gen_neg_1vN(dm, N_1vN, rest_thres)
                iMvN.append([neg_1vN[0]+1, list(np.array(neg_1vN[1])+1)])
            iMvN.append(random.choice(range(1, posnum_1v1[i]+1)))
            rest.append(iMvN)

    if save_path is not None:
        joblib.dump(rest, save_path)
    else:
        print(f'The sampled {rest_type} restraints:\n{rest}')
        
        



    

