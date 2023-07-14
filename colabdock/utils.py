import os
import numpy as np


def prep_path(save_path):
  if not os.path.exists(f'{save_path}/gen'):
    os.makedirs(f'{save_path}/gen')
  if not os.path.exists(f'{save_path}/pred'):
    os.makedirs(f'{save_path}/pred')
  if not os.path.exists(f'{save_path}/docked'):
    os.makedirs(f'{save_path}/docked')


def renum_pdb(inpdb_path, outpdb_path, ind2ID):
  # delete gap & change chain id
  # renumber atom id & chain id (todo)
  with open(inpdb_path, 'r') as f:
    lines = f.read().splitlines()
  with open(outpdb_path, 'w') as f:
    for iline in lines:
      if len(iline) > 50 and iline.startswith('ATOM'):
        chainID = ind2ID[int(iline[22:26])]
        f.write(f"{iline[:21]}{chainID}{iline[22:]}\n")


def inflate(mask, crop_len):
    def get_segs(arr):
        idx = np.where(arr == 1)[0]
        segs, iseg = [], []
        for ith, ind in enumerate(idx):
            if not iseg:
                iseg.append(ind)
                count = ind
            elif ind != count + 1:
                iseg.append(idx[ith - 1])
                segs.append(iseg)
                iseg = [ind]
                count = ind
            else:
                count += 1
        if iseg:
            iseg.append(idx[-1])
            segs.append(iseg)
        return np.array(segs)


    def expand_segs(segs, mask, add_num):
        break_flag = 0
        if type(add_num) == int:
            iadd_num = add_num
        else:
            assert type(add_num) == list
            assert len(segs) == len(add_num)

        for ith, iseg in enumerate(segs):
            if type(add_num) == list:
                iadd_num = add_num[ith]
            iseg_start = iseg[0]
            start = max([0, iseg_start - iadd_num])
            mask[start:iseg_start] = 1
            if mask.sum() == crop_len:
                break_flag = 1
                break

            iseg_stop = iseg[1]
            stop = min([iseg_stop + iadd_num + 1, L])
            mask[iseg_stop + 1:stop] = 1
            if mask.sum() == crop_len:
                break_flag = 1
                break
        return break_flag, mask
    
    L = len(mask)
    if crop_len is None:
        mask = np.ones(L, dtype=np.int32)
    elif mask.sum() > crop_len:
        raise Exception('too many restraint amino acids are provided!')
    elif mask.sum() < crop_len:
        break_flag = 0
        segs = get_segs(mask)
        seg_lens = [iseg[1] - iseg[0] + 1 for iseg in segs]
        ave_len = crop_len // len(segs)
        add_num = [max(ave_len - iseg_len, 0) // 2 for iseg_len in seg_lens]
        while True:
            break_flag, mask = expand_segs(segs, mask, add_num)
            add_num = max([1, int((crop_len - mask.sum()) // (len(segs) * 2))])
            segs = get_segs(mask)
            if break_flag == 1:
                break

        assert mask.sum() == crop_len
    return mask
