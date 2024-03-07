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
    template_r = config.template
    native_r = config.native
    chains = config.chains
    template = {'pdb_path': template_r,
                'chains': chains}
    native = {'pdb_path': native_r,
              'chains': chains}
    fixed_chains = config.fixed_chains

    ######################################################################################
    # experimental restraints
    ######################################################################################
    rest_MvN_r = config.rest_MvN
    rest_non_r = config.rest_rep
    rest_1vN_r = config.rest_1vN
    rest_1v1_r = config.rest_1v1

    # print restraints
    print_str = f'restraints:\n'
    if rest_1v1_r is None:
        print_str += '\tno 1v1 restraints provided.\n'
    else:
        print_str += f'\t1v1 restraints:\n\t\t{rest_1v1_r}\n'
    
    if rest_1vN_r is None:
        print_str += '\tno 1vN restraints provided.\n'
    else:
        print_str += f'\t1vN restraints:\n\t\t{rest_1vN_r}\n'
    
    if rest_MvN_r is None:
        print_str += '\tno MvN restraints provided.\n'
    else:
        print_str += f'\tMvN restraints:\n\t\t{rest_MvN_r}\n'
    
    if rest_non_r is None:
        print_str += '\tno repulsive restraints provided.\n'
    else:
        print_str += f'\trepulsive restraints:\n\t\t{rest_non_r}\n'

    # 1v1
    if rest_1v1_r is not None:
        if type(rest_1v1_r[0]) is not list:
            rest_1v1_r = [rest_1v1_r]
        rest_1v1 = np.array(rest_1v1_r) - 1
    else:
        rest_1v1 = None
    
    # 1vN
    if rest_1vN_r is not None:
        if type(rest_1vN_r[0]) is not list:
            rest_1vN_r = [rest_1vN_r]
        rest_1vN = []
        for irest_1vN in rest_1vN_r:
            rest_1vN.append([irest_1vN[0] - 1, np.array(irest_1vN[1]) - 1])
    else:
        rest_1vN = None
    
    # MvN
    if rest_MvN_r is not None:
        if type(rest_MvN_r[-1]) is not list:
            rest_MvN_r = [rest_MvN_r]
        rest_MvN = []
        for irest_MvN in rest_MvN_r:
            irest = []
            for irest_1vN in irest_MvN[:-1]:
                irest.append([irest_1vN[0] - 1, np.array(irest_1vN[1]) - 1])
            irest.append(irest_MvN[-1])
            rest_MvN.append(irest)
    else:
        rest_MvN = None
    
    # repulsive
    if rest_non_r is not None:
        if type(rest_non_r[0]) is not list:
            rest_non_r = [rest_non_r]
        rest_non = np.array(rest_non_r) - 1
    else:
        rest_non = None

    restraints = {'1v1': rest_1v1,
                  '1vN': rest_1vN,
                  'MvN': rest_MvN,
                  'non': rest_non}
    
    res_thres = config.res_thres
    non_thres = config.rep_thres

    ######################################################################################
    # optimization parameters
    ######################################################################################
    rounds = config.rounds
    crop_len = config.crop_len
    step_num = config.steps
    save_every_n_step = config.save_every_n_step
    data_dir = config.data_dir
    bfloat = config.bfloat
    use_multimer = config.use_multimer

    ######################################################################################
    # print setting
    ######################################################################################
    print_str += '\nOptimization losses include:\n\t'
    if rest_1v1 is not None:
        print_str += '1v1 restraint loss, '
    if rest_1vN is not None:
        print_str += '1vN restraint loss, '
    if rest_MvN is not None:
        print_str += 'MvN restraint loss, '
    if rest_non is not None:
        print_str += 'repulsive restraint loss, '
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
                           save_every_n_step=save_every_n_step,
                           use_multimer=use_multimer)
    dock_model.setup()
    if dock_model.crop_len is not None:
        print_str += 'Colabdock will work in segment based mode.'
    print(print_str)
    print('\nStart optimization')
    dock_model.dock_rank()
