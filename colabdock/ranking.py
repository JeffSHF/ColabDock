import numpy as np
import joblib

from colabdesign.af.prep import prep_pdb
from colabdesign.af.loss import _get_rmsd_loss
from colabdesign.af.alphafold.model import model
from colabdock.utils import renum_pdb


class _rank:
    def __init__(self):
        self.model_inter = joblib.load('./colabdock/weights_rank/inter.pkl')
        self.model_intra = joblib.load('./colabdock/weights_rank/intra.pkl')

    def rank_fea(self, iepoch):
        feature = []
        rest_num = self.rest_num
        for i in range(len(self.gen_measures[iepoch])):
            #################################
            # prediction stage
            #################################
            save_path = f'{self.save_path}/pred/pred_{iepoch+1}_{i+1}.pdb'
            (dis_rmsd, dis_satis_num, dis_dist_error, dis_satis_num_all, 
             dis_dist_error_all, dis_avg_plddt, dis_con_num) = self._cal_rfea(save_path, self.dis_measures[iepoch][i][1])
            dis_iptm = self.dis_measures[iepoch][i][0]
            feature.append([dis_con_num, dis_avg_plddt, dis_iptm,
                            dis_satis_num_all/rest_num,
                            dis_dist_error_all/rest_num,
                            dis_rmsd, dis_satis_num, dis_dist_error])
        return np.array(feature)

    def _cal_rfea(self, save_path, plddt):
        pdb_pred = prep_pdb(save_path)
        renum_pdb(save_path, save_path, self.ind2ID)
        # cal rmsd
        if self.gt_obj is not None:
            rmsd = _get_rmsd_loss(self.gt_obj["batch"]["all_atom_positions"][:,1,:],
                                  pdb_pred["batch"]["all_atom_positions"][:,1,:])
            rmsd = float(rmsd['rmsd'])
        else:
            rmsd = None
        # cal performance
        pb_pred, _ = model.modules.pseudo_beta_fn(pdb_pred['batch']['aatype'],
                                                  pdb_pred['batch']["all_atom_positions"],
                                                  pdb_pred['batch']["all_atom_mask"])
        dist = np.sqrt(np.square(pb_pred[:, None] - pb_pred[None, :]).sum(-1) + 1e-8)
        rest_dict = {'1v1': {'satis_num': None,
                             'dist_error': None},
                     '1vN': {'satis_num': None,
                             'dist_error': None},
                     'MvN': {'satis_num': None,
                             'dist_error': None},
                     'non': {'satis_num': None,
                             'dist_error': None},}
        # 1v1 restraints
        rest_1v1 = self.rest_raw['1v1']
        if rest_1v1 is not None:
            satis_num = sum([1 if dist[res1, res2] < self.res_thres else 0 for res1, res2 in rest_1v1])
            dist_error = sum([max(dist[res1, res2] - self.res_thres, 0) for res1, res2 in rest_1v1])
            rest_dict['1v1']['satis_num'] = satis_num
            rest_dict['1v1']['dist_error'] = dist_error
        else:
            rest_dict['1v1']['satis_num'] = 0
            rest_dict['1v1']['dist_error'] = 0

        # 1vN restraints
        rest_1vN = self.rest_raw['1vN']
        if rest_1vN is not None:
            satis_num = sum([1 if dist[ires, tuple(icdr)].min() < self.res_thres else 0 for ires, icdr in rest_1vN])
            dist_error = sum([max(dist[ires, tuple(icdr)].min() - self.res_thres, 0) for ires, icdr in rest_1vN])
            rest_dict['1vN']['satis_num'] = satis_num
            rest_dict['1vN']['dist_error'] = dist_error
        else:
            rest_dict['1vN']['satis_num'] = 0
            rest_dict['1vN']['dist_error'] = 0

        # MvN restraints
        rest_MvN = self.rest_raw['MvN']
        if rest_MvN is not None:
            satis_num, dist_error = [], []
            satis_num_all, dist_error_all = [], []
            for irest_MvN in rest_MvN:
                rest_1vN, topk = irest_MvN[:-1], irest_MvN[-1]
                inum = [1 if dist[ires, tuple(icdr)].min() < self.res_thres else 0 for ires, icdr in rest_1vN]
                inum = np.sort(np.array(inum))[::-1]
                satis_num.append(inum[:topk].sum())
                satis_num_all.append(inum.sum())

                ierror = [max(dist[ires, tuple(icdr)].min() - self.res_thres, 0) for ires, icdr in rest_1vN]
                ierror = np.sort(np.array(ierror))
                dist_error.append(ierror[:topk].sum())
                dist_error_all.append(ierror.sum())
            rest_dict['MvN']['satis_num'] = [satis_num, satis_num_all]
            rest_dict['MvN']['dist_error'] = [dist_error, dist_error_all]
        else:
            rest_dict['MvN']['satis_num'] = [0, 0]
            rest_dict['MvN']['dist_error'] = [0, 0]

        # non restraints
        rest_non = self.rest_raw['non']
        if rest_non is not None:
            satis_num = sum([1 if dist[res1, res2] > self.non_thres else 0 for res1, res2 in rest_non])
            dist_error = sum([max(self.non_thres - dist[res1, res2], 0) for res1, res2 in rest_non])
            rest_dict['non']['satis_num'] = satis_num
            rest_dict['non']['dist_error'] = dist_error
        else:
            rest_dict['non']['satis_num'] = 0
            rest_dict['non']['dist_error'] = 0    

        satis_num, dist_error = 0, 0
        satis_num_all, dist_error_all = 0, 0
        for ik, iv in rest_dict.items():
            for iik, iiv in iv.items():
                if iik == 'satis_num' and iiv is not None:
                    if ik == 'MvN':
                        satis_num += np.array(iiv[0]).sum()
                        satis_num_all += np.array(iiv[1]).sum()
                    else:
                        satis_num += iiv
                        satis_num_all += iiv
                if iik == 'dist_error' and iiv is not None:
                    if ik == 'MvN':
                        dist_error += np.array(iiv[0]).sum()
                        dist_error_all += np.array(iiv[1]).sum()
                    else:
                        dist_error += iiv
                        dist_error_all += iiv

        mask_inter = self.asym_id[:, None] != self.asym_id[None, :]
        dist_inter = dist * mask_inter
        x, y = np.where((dist_inter <= 8.0) * (dist_inter > 0.0) == 1)
        
        # interface_contacts_number and avg_interface_plddt
        con_num = len(x) / 2 + 1e-8
        idx = list(set(list(x) + list(y)))
        avg_plddt = np.mean(plddt[idx]) if idx else 1e-8

        return rmsd, satis_num, dist_error, satis_num_all, dist_error_all, avg_plddt, con_num


    @staticmethod
    def rank_struct(rank_model, feature, topk=5):
        score = []
        for i in range(len(feature)):
            t = []
            for k in range(len(feature)):
                if i!=k:
                    t.append(feature[i] - feature[k])
            score.append(sum(rank_model.predict(np.array(t))))
        idx = [(i, score[i]) for i in range(len(score))]
        idx.sort(key=lambda x:x[1], reverse=True)
        idx = np.array(idx)
        return np.array(idx[:topk, 0], dtype=np.int32)
