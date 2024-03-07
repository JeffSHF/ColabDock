import numpy as np
import os

from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.common import residue_constants

from colabdock.docking import _dock
from colabdock.ranking import _rank
from colabdock.prep import _rest

class ColabDock(_dock, _rank, _rest):
    def __init__(self,
                 template,
                 restraints,
                 save_path,
                 data_dir,
                 msa_path=None,
                 structure_gt=None,
                 crop_len=None,
                 fixed_chains=None,
                 round_num=2,
                 step_num=50,
                 prob_rest=0.5,
                 bfloat=True,
                 res_thres=8.0,
                 non_thres=12.0,
                 save_every_n_step=1,
                 use_multimer=False) -> None:
        self.template = template
        self.structure_gt = structure_gt
        self.fixed_chains = fixed_chains

        self.rest_raw = restraints
        self.res_thres = res_thres
        self.non_thres = non_thres

        self.step_num = step_num
        self.round_num = round_num
        self.crop_len = crop_len
        self.prob_rest = prob_rest
        self.bfloat = bfloat
        self.use_multimer=use_multimer

        self.save_path = save_path
        self.data_dir = data_dir
        self.save_every_n_step = save_every_n_step

        self.use_initial = True
        self.split_templates = False
        self.use_dgram = True
        self.msas = msa_path
        self.rm_template_seq = False

        self.w_non = 1.0
        self.w_res = 2.0
        self.lr = 0.1
    
    def setup(self):
        # process the input structures
        if self.structure_gt['pdb_path'] is not None:
            self.gt_obj = prep_pdb(self.structure_gt['pdb_path'],
                                   chain=self.structure_gt['chains'],
                                   for_alphafold=False)
        else:
            self.gt_obj = None
        
        tmp_obj = prep_pdb(self.template['pdb_path'],
                           chain=self.template['chains'],
                           for_alphafold=False)    
        self.seq_wt = ''.join([residue_constants.restypes[ind] for ind in tmp_obj['batch']['aatype']])

        if self.crop_len is not None:
            if self.crop_len >= len(self.seq_wt):
                self.crop_len = None
        
        self.lens = np.array([np.where(tmp_obj['idx']['chain']==ichain)[0].size for ichain in self.template['chains'].split(',')])
        chains = self.template['chains'].split(',')
        if self.fixed_chains is None:
            self.fixed_chains = self.template['chains'].split(',')

        lens_cumsum = [0] + list(np.cumsum(self.lens))
        self.asym_id = np.ones(self.lens.sum())
        for ith, icomp in enumerate(self.fixed_chains):
            for ichain in icomp.split(','):
                assert ichain in chains
                ind = chains.index(ichain)
                self.asym_id[lens_cumsum[ind]:lens_cumsum[ind+1]] *= ith

        # var for correcting the chainID in saved pdb
        self.ind2ID = np.array(['-'] * (self.lens.sum() + (len(chains) - 1) * 50 + 1))
        bounds = [0] + list(np.cumsum(self.lens + np.array([50] * len(chains))))
        bounds = np.array(bounds, dtype=np.int32) + 1
        for ith, ichain in enumerate(chains):
            self.ind2ID[bounds[ith]:bounds[ith + 1]] = ichain
        
        # process the raw restraints
        self.process_restraints(self.rest_raw)

        # initial colabdesign
        _dock.__init__(self)

        # initial ranking model
        _rank.__init__(self)
    
    def dock_rank(self):
        ######################################################################################
        # dock
        ######################################################################################
        for ith in range(self.round_num):
            self.optimize(ith)
        self.inference()

        ######################################################################################
        # rank
        ######################################################################################
        # rank within each round
        feature_topk, idx_topk = [], []
        for ith in range(self.round_num):
            feature = self.rank_fea(ith)
            sel_idx = self.rank_struct(self.model_intra, feature[:, :5])
            feature_topk.extend(feature[sel_idx, :])
            idx_topk.extend([[ith, ind] for ind in sel_idx])
        
        # rank between rounds
        feature_topk = np.array(feature_topk)
        sel_idx = self.rank_struct(self.model_inter, feature_topk[:, :5])

        # save top5 structures
        with open(f'{self.save_path}/docked/summary.txt', 'w') as f:
            f.write('pdb\t')
            if self.gt_obj is not None:
                f.write('RMSD\t')
            f.write('iptm\t# of satisfied restraints\n')
            for ith, ind in enumerate(sel_idx):
                iepoch, istep = idx_topk[ind]
                if ith == 0:
                    pdbname = '1st_best'
                elif ith == 1:
                    pdbname = '2nd_best'
                elif ith == 2:
                    pdbname = '3rd_best'
                elif ith == 3:
                    pdbname = '4th_best'
                elif ith == 4:
                    pdbname = '5th_best'
                comm = f'cp {self.save_path}/pred/pred_{iepoch+1}_{istep+1}.pdb {self.save_path}/docked/{pdbname}.pdb'
                os.system(comm)

                _, _, dis_iptm, _, _, dis_rmsd, dis_satis_num, _ = feature_topk[ind]
                print_str = f'{pdbname} structure:\n\t'
                f.write(f'{pdbname}.pdb\t')
                if dis_rmsd is not None:
                    print_str += f'rmsd: {dis_rmsd:.3f}, '
                    f.write(f'{dis_rmsd:.3f}\t')
                print_str += (f'iptm: {dis_iptm:.3f}, {int(dis_satis_num):d} out of {int(self.rest_num)} restraints are satisfied.')
                print(print_str)
                f.write(f'{dis_iptm:.3f}\t{int(dis_satis_num)}/{int(self.rest_num)}\n')
