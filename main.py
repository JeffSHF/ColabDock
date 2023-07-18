import os
import numpy as np
from config import config
from colabdock.utils import prep_path
from colabdock.model import ColabDock

np.set_printoptions(precision=3)


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
if __name__ == '__main__':
    save_path = config.save_path
    prep_path(save_path)
    ######################################################################################
    # template and native structure
    ######################################################################################
    template = config.template
    native = config.native
    fixed_chains = config.fixed_chains

    ######################################################################################
    # experimental restraints
    ######################################################################################
    rest_MvN = config.rest_MvN
    rest_non = config.rest_non
    rest_1vN = config.rest_1vN
    rest_1v1 = config.rest_1v1

    restraints = {'1v1': rest_1v1,
                  '1vN': rest_1vN,
                  'MvN': rest_MvN,
                  'non': rest_non}
    
    res_thres = config.res_thres
    non_thres = config.non_thres

    ######################################################################################
    # optimization parameters
    ######################################################################################
    rounds = config.rounds
    crop_len = config.crop_len
    step_num = config.steps
    save_every_n_step = config.save_every_n_step
    data_dir = config.data_dir
    bfloat = config.bfloat

    ######################################################################################
    # print setting
    ######################################################################################
    print_str = f'restraints:\n'
    if rest_1v1 is None:
        print_str += '\tno 1v1 restraints provided.\n'
    else:
        print_str += f'\t1v1 restraints:\n\t\t{rest_1v1}\n'
    
    if rest_1vN is None:
        print_str += '\tno 1vN restraints provided.\n'
    else:
        print_str += f'\t1vN restraints:\n\t\t{rest_1vN}\n'
    
    if rest_MvN is None:
        print_str += '\tno MvN restraints provided.\n'
    else:
        print_str += f'\tMvN restraints:\n\t\t{rest_MvN}\n'
    
    if rest_non is None:
        print_str += '\tno non restraints provided.\n'
    else:
        print_str += f'\tnon restraints:\n\t\t{rest_MvN}\n'
    
    print_str += '\nOptimization losses include:\n\t'
    if rest_1v1 is not None:
        print_str += '1v1 restraint loss, '
    if rest_1vN is not None:
        print_str += '1vN restraint loss, '
    if rest_MvN is not None:
        print_str += 'MvN restraint loss, '
    if rest_non is not None:
        print_str += 'non restraint loss, '
    print_str += 'distogram loss, pLDDT, and ipAE.\n'

    ######################################################################################
    # start docking
    ######################################################################################
    dock_model = ColabDock(template,
                           restraints,
                           save_path,
                           data_dir,
                           structure_gt=native,
                           crop_len=crop_len,
                           fixed_chains=fixed_chains,
                           round_num=rounds,
                           step_num=step_num,
                           bfloat=bfloat,
                           res_thres=res_thres,
                           non_thres=non_thres,
                           save_every_n_step=save_every_n_step)
    dock_model.setup()
    if dock_model.crop_len is not None:
        print_str += 'Colabdock will work in segment based mode.'
    print(print_str)
    print('\nStart optimization')
    dock_model.dock_rank()
