import ml_collections
import joblib


config = {
    # path where you want to save the results
    'save_path': './results',

    ###########################################################################################################
    # template and native structure
    ###########################################################################################################
    # the structure of proteins you want to dock
    'template': './protein/4INS4/PDB/template.pdb',

    # optional, the native structure of the complex, used for calculating the RMSD and DockQ
    # if not provided, set to None
    'native': './protein/4INS4/PDB/native.pdb',

    # docking chains
    # This determines the order that the chain sequences are concatenated to form the complex sequence.
    'chains': 'A,B,C,D',

    # input the chainIDs if you want the relative position of chains is fixed as in the provided template
    # otherwise, set to None
    # example:
    #     'fixed_chains': ['A,B', 'C,D']
    #     the relative position of chain A and B is fixed, also that of chain C and D.
    'fixed_chains': ['A,B', 'C,D'],

    ###########################################################################################################
    # experimental restraints
    ###########################################################################################################
    # the threshold of the experimental restraints, usually set to 8.0Å.
    # Change to other values if you know the threshold of the restraints you provide.
    # Due to the definition of distogram in AF2, threshold should be set to a value between 2Å and 22Å
    'res_thres': 8.0,

    # 1v1 restraints
    # description:
    #     The distance between two residues is below a given threshold (res_thres).
    #     If there is no such restraints, set to None.
    #     If you have multiple 1v1 restraints, list them in a [].
    #     The order number in a 1v1 restraint refers to the residue in the complex sequence.
    #         The complex sequence is concatenated by the chain sequences and the order is determined by the "chains" provided above.
    #         This is the same for the remaining types of restraints.
    #     The order number starts from 1.
    # example:
    #     'rest_1v1': [[78,198],[20,50]]
    #     The distance between 78th and 198th residue is below a given threshold, as well as the distance between 20th and 50th residue.
    'rest_1v1': None,

    # 1vN restraints
    # description:
    #     the distance between one residue and a residue set is below a given threshold (res_thres)
    #     if there is no such restraints, set to None
    # example:
    #     'rest_1vN': [[36, np.array(range(160, 170))]]
    #     the distance between 36th residue and at least a residue from 160th to 170th is below a given threshold
    #     the order number starts from 0
    'rest_1vN': None,

    # MvN restraints
    # description:
    #     contain several 1vN restraints, and only a specific number of them are satisfied
    #     if there is no such restraints, set to None
    # example:
    #     'rest_MvN': [[[10, np.array(range(160, 170))],
    #                   [78, np.array(range(160, 170))],
    #                   [120, np.array(range(160, 170))],
    #                  2]]
    #     2 of the 3 given 1vN restraints should be satisfied
    #     the order number starts from 0
    'rest_MvN': joblib.load('./protein/4INS4/rest_MvN.pkl'),
    
    # the threshold of the non restraints
    # Change to other values if you know the threshold of the restraints you provide.
    # Due to the definition of distogram in AF2, threshold should be set to a value between 2Å and 22Å
    'non_thres': 8.0,

    # non restraints
    # description:
    #     the distance between two residues is above a given threshold (non_thres)
    #     if there is no such restraints, set to None
    # example:
    #     'rest_non: np.array([[154, 250]])
    #     the distance between 154th and 250th residue is above a given threshold
    #     the order number starts from 0
    'rest_non': None,

    ###########################################################################################################
    # optimization parameters
    ###########################################################################################################
    # if in segment based optimization, set to the length of the segment, for example 200.
    # segment based optimization can save GPU memory, but may lead to suboptimal performance.
    # if not, set to None
    'crop_len': None,

    # the number of rounds to perform
    # large rounds can achive better performance but lead to longer time.
    'rounds': 2,

    # the number of backpropogations in each round
    # if in segment based optimization, set to larger value, for example 150.
    # if not, usually it will converge within 50 steps
    'steps': 50,

    # Save one conformtion in every save_every_n_step step.
    # useful in segment based optimization, since the number of steps is larger
    # and saving conformations in every step will take too much time.
    # if in segment based optimization, set to larger value, for example 3.
    # if not, set to 1.
    'save_every_n_step': 1,

    ###########################################################################################################
    # AF2 model
    ###########################################################################################################
    # AF2 weights dir
    'data_dir': '/data/dell/alphafold',

    # whether use AF2 in bfloat
    'bfloat': True,
}

config = ml_collections.ConfigDict(config)
