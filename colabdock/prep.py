import numpy as np
from colabdock.utils import inflate

class _rest:
    def process_restraints(self, rest_raw):
        rest_1v1 = rest_raw['1v1']
        rest_1vN = rest_raw['1vN']
        rest_MvN = rest_raw['MvN']
        rest_non = rest_raw['non']

        # assert not (rest_1v1 is None and rest_1vN is None and rest_MvN is None)
        # the indices of residues cropped out
        restraints_p = {'1v1': None,
                        '1vN': None,
                        'MvN': None,
                        'non': None}
        # mask
        restraints_mask = {'1v1': None,
                           '1vN': None,
                           'MvN': None,
                           'non': None}
        rest_MvN_num = np.array([1])
        L = len(self.seq_wt)

        rest_num = 0
        # check 1v1 restraints
        if rest_1v1 is not None:
            assert len(rest_1v1.shape) == 2 and rest_1v1.shape[1] == 2
            mask_1d = np.zeros(L)
            mask_2d = np.zeros([L, L])
            for ires in rest_1v1:
                mask_1d[ires[0]] = 1
                mask_1d[ires[1]] = 1
                mask_2d[ires[0], ires[1]] = 1
            mask_1d = inflate(mask_1d, self.crop_len)
            mask_2d += mask_2d.T
            mask_2d = np.where(mask_2d == 0, 0, 1)
            restraints_p['1v1'] = np.where(mask_1d == 1)[0]
            restraints_mask['1v1'] = mask_2d
            rest_num += len(rest_1v1)

        # check 1vN restraints
        # TODO: each time optimize one 1vN restraint, maybe not that necessary
        if rest_1vN is not None:
            mask_1d = np.zeros(L)
            mask_2d = np.zeros([L, L])
            for irest_1vN in rest_1vN:
                assert type(irest_1vN[0]) == int
                mask_1d[irest_1vN[0]] = 1
                assert type(irest_1vN[1]) == np.ndarray
                mask_1d[irest_1vN[1]] = 1
                mask_2d[irest_1vN[0], irest_1vN[1]] = 1
            mask_1d = inflate(mask_1d, self.crop_len)
            mask_2d = np.where(mask_2d == 0, 0, 1)
            restraints_p['1vN'] = np.where(mask_1d == 1)[0]
            restraints_mask['1vN'] = mask_2d
            rest_num += len(rest_1vN)

        # check MvN restraints
        if rest_MvN is not None:
            mask_1d, mask_2d, satis_num = [], [], []
            for irest_MvN in rest_MvN:
                imask_1d = np.zeros(L)
                imask_2d = np.zeros([L, L])
                assert type(irest_MvN[-1]) == int
                for irest_1vN in irest_MvN[:-1]:
                    assert type(irest_1vN[0]) == int or type(irest_1vN[0]) == np.int64
                    imask_1d[irest_1vN[0]] = 1
                    assert type(irest_1vN[1]) == np.ndarray
                    imask_1d[irest_1vN[1]] = 1
                    imask_2d[irest_1vN[0], irest_1vN[1]] = 1
                imask_1d = inflate(imask_1d, self.crop_len)
                mask_1d.append(imask_1d)
                imask_2d = np.where(imask_2d == 0, 0, 1)
                mask_2d.append(imask_2d)
                satis_num.append(int(irest_MvN[-1]))
            restraints_p['MvN'] = np.array([np.where(imask_1d == 1)[0] for imask_1d in mask_1d])
            restraints_mask['MvN'] = np.array(mask_2d)
            rest_MvN_num = np.array(satis_num)
            rest_num += rest_MvN_num.sum()

        # check non restraints
        if rest_non is not None:
            mask_1d = np.zeros(L)
            mask_2d = np.zeros([L, L])
            assert len(rest_non.shape) == 2 and rest_non.shape[1] == 2
            for ires in rest_non:
                mask_1d[ires[0]] = 1
                mask_1d[ires[1]] = 1
                mask_2d[ires[0], ires[1]] = 1
            mask_1d = inflate(mask_1d, self.crop_len)
            mask_2d += mask_2d.T
            mask_2d = np.where(mask_2d == 0, 0, 1)
            mask_2d = np.triu(mask_2d)
            restraints_p['non'] = np.where(mask_1d == 1)[0]
            restraints_mask['non'] = mask_2d
            rest_num += len(rest_non)

        self.rest_set = {'rest_p': restraints_p,
                         'rest_mask': restraints_mask,
                         'rest_MvN_num': rest_MvN_num}
        
        self.rest_num = rest_num + 1e-8
        
        if self.crop_len is not None:
            sample_p = [rest_1v1, rest_1vN, rest_MvN, rest_non]
            sample_p = [0 if ires is None else 1 for ires in sample_p]
            self.flag_valid = sample_p
            sample_p = np.array(sample_p)
            sample_p = sample_p / (sample_p.sum() + 1e-8)
            self.sample_p = sample_p.cumsum() * self.prob_rest
