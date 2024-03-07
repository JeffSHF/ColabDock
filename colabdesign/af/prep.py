import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R
from Bio import pairwise2

from colabdesign.af.alphafold.data import pipeline, prep_inputs
from colabdesign.af.alphafold.common import protein, residue_constants
from colabdesign.af.alphafold.model import all_atom, model
from colabdesign.af.alphafold.model.tf import shape_placeholders
from colabdesign.af.inputs import update_aatype
from colabdesign.af.utils import read_msa

from colabdesign.shared.protein import _np_get_cb, pdb_to_string
from colabdesign.shared.prep import prep_pos
from colabdesign.shared.utils import copy_dict
from colabdesign.shared.model import order_aa

resname_to_idx = residue_constants.resname_to_idx
idx_to_resname = dict((v,k) for k,v in resname_to_idx.items())

#################################################
# AF_PREP - input prep functions
#################################################
class _af_prep:
  # prep functions specific to protocol
  def _prep_dock(self, rest_set, template, fixed_chains=None, chain_weights=None,
                 use_initial=True, msas=None, copies=1, repeat=False,
                 block_diag=False, split_templates=False,
                 use_dgram=True, rm_template_seq=True, use_multimer=False, **kwargs):
    '''prep inputs for docking'''
    assert not (template is None and msas is None)

    self.rest_set = rest_set
    self.split_templates = split_templates
    self.use_dgram = use_dgram
    self._args["rm_template_seq"] = rm_template_seq
    
    # block_diag the msa features
    if block_diag and copies > 1:
      self._num *= (1 + copies)
    else:
      block_diag = False
    self._args.update({"block_diag":block_diag, "repeat":repeat})
    
    self.opt["weights"].update({"con":0.0, "plddt":0.0, "pae":0.0, "helix": 0.0,
                                "dgram_cce":0.0, "rest_non":0.0,
                                "rest_1v1": 0.0, "rest_1vN": 0.0, "rest_MvN": 0.0,
                                "i_pae":0.0, "i_con":0.0})

    # process templates
    pdb = prep_pdb(template['pdb_path'], chain=template['chains'])
    self._batch = pdb['batch']
    chains = [c.strip() for c in template['chains'].split(",")]
    self.lens = [(pdb["idx"]["chain"] == c).sum() for c in chains]
    self._len = sum(self.lens)
    self._wt_aatype = self._batch["aatype"]
    boundaries = [0] + list(np.cumsum(self.lens))

    # msa related
    wt_seq = ''.join([residue_constants.restypes[iaa] for iaa in self._wt_aatype])
    if msas is not None:
      self._args["use_msa"] = True
      input_msa, input_dlm = read_msa(msas)
      assert len(input_msa) >= 1
      assert len(input_msa[0]) == self._len
      num_msa = min(len(input_msa), self.opt['num_msa'])
      input_msa, input_dlm = input_msa[:num_msa], input_dlm[:num_msa]
      if wt_seq != input_msa[0]:
        input_msa = [wt_seq] + input_msa[:-1]
        input_dlm = [[0] * self._len] + input_dlm[:-1]
    else:
      num_msa = 1

    # process fixed_chains
    self.fixed_chains = fixed_chains
    if self.fixed_chains is not None:
      cliques = []
      for icomp in self.fixed_chains:
        icomp = [ichain.strip() for ichain in icomp.split(',')]
        icomp = [chains.index(ichain) for ichain in icomp]
        cliques.append(icomp)
      # check
      cliques_chains = [ichain for iclique in cliques for ichain in iclique]
      if len(cliques_chains) != len(set(cliques_chains)):
        raise Exception('Provided fixed chains have overlap!')
      for i in range(len(chains)):
        if i not in cliques_chains:
          cliques.append([i])
    else:
      cliques = [[i] for i in range(len(chains))]
    self.cliques = cliques

    # seq init
    seqs = []
    for ind in range(len(chains)):
      start, stop = boundaries[ind], boundaries[ind+1]
      seqs.append(wt_seq[start:stop])

    sim_mat = np.ones([len(chains), len(chains)])
    for ith in range(len(chains)):
      ilen = len(seqs[ith])
      for jth in range(ith+1, len(chains)):
        jlen = len(seqs[jth])
        if min(ilen, jlen) / max(ilen, jlen) > 0.8:
          alignments = pairwise2.align.globalxx(seqs[ith], seqs[jth])
          if cal_seqid(alignments[0]) > 0.8:
            sim_mat[ith, jth] = 0
    
    del_idx = []
    for ith in range(1, len(chains)):
      if not (all(sim_mat[ith]) and all(sim_mat[:,ith])):
        del_idx.append(ith)
        sim_mat[ith] = 1
        sim_mat[:, ith] = 1
    
    init_mask = np.ones(self._len)
    for ind in del_idx:
      start, stop = boundaries[ind], boundaries[ind+1]
      init_mask[start:stop] = 0

    noise_mask = np.random.rand(self._len)
    noise_mask = np.where(noise_mask > 0.7, 1, 0)
    init_mask += noise_mask*(1. - init_mask)
    
    seq_init = np.eye(20)[self._wt_aatype] * init_mask[..., None]
    seq_init += np.random.normal(size=(self._len, 20)) * 0.1 * (1. - init_mask[..., None])

    self.seq_init = seq_init

    # generate initial position
    if use_initial:
      assert len(cliques) <= 8
      pdb_pos = pdb['batch']['all_atom_positions']
      pdb_mask = pdb['batch']['all_atom_mask']

      # generate rotation matrices
      mat_rot = [R.random() for _ in range(len(cliques))]

      # this is used only in development!
      pdb_chains = {}
      for ind in range(len(chains)):
        start, stop = boundaries[ind], boundaries[ind+1]
        ipdb_pos = pdb_pos[start:stop]
        pdb_chains[ind] = ipdb_pos
      
      # rotate randomly
      for ind, iclique in enumerate(cliques):
        for ichain in iclique:
          ipdb_pos = pdb_chains[ichain]
          ipdb_pos = mat_rot[ind].apply(ipdb_pos.reshape([-1, 3])).reshape([self.lens[ichain], 37, 3])
          pdb_chains[ichain] = ipdb_pos
      
      pdb_max, pdb_min = [], []
      for ind, iclique in enumerate(cliques):
        coord = []
        for ichain in iclique:
          start, stop = boundaries[ichain], boundaries[ichain+1]
          for ith, ith_a in enumerate(range(start, stop)):
            for ia in range(37):
              if pdb_mask[ith_a, ia] == 1:
                coord.append(pdb_chains[ichain][ith][ia])
        coord = np.array(coord)
        pdb_max.append(np.max(coord, 0))
        pdb_min.append(np.min(coord, 0))

      for ind, iclique in enumerate(cliques):
        code_sub = ['0', '0', '0']
        str_sub = bin(ind).replace('0b', '')
        code_sub[-len(str_sub):] = list(str_sub)
        for ichain in iclique:
          pdb_chain_sub = []
          for ipos in range(3):
            if code_sub[ipos] == '1':
              pdb_chain_sub.append(pdb_chains[ichain][:, :, ipos] - pdb_min[ind][ipos] + 5.0)
            else:
              pdb_chain_sub.append(pdb_chains[ichain][:, :, ipos] - pdb_max[ind][ipos] - 5.0)
          pdb_chain_sub = np.stack(pdb_chain_sub, -1)
          pdb_chains[ichain] = pdb_chain_sub
      
      # concatenate all coord
      pdb_initial = []
      for ind in range(len(chains)):
        pdb_initial.append(pdb_chains[ind])
      pdb_initial = np.concatenate(pdb_initial, 0)

    num_templates = len(cliques) if split_templates and not use_dgram else 1
    # add another zero template to avoid extra compile
    num_templates = num_templates + 1
    self._inputs = prep_input_features(L=self._len, N=num_msa, T=num_templates, eN=1)
    self._inputs = jax.tree_map(lambda x:jnp.array(x), self._inputs)

    # input generated initial position
    L = self._inputs["residue_index"].shape[-1]
    if not use_initial:
      pdb_initial = np.zeros([L,37,3])
    self._inputs["prev"] = {'prev_msa_first_row': np.zeros([L,256]),
                            'prev_pair': np.zeros([L,L,128]),
                            'prev_pos': pdb_initial}

    # template distance matrix
    x_beta, _ = model.modules.pseudo_beta_fn(aatype=pdb['batch']['aatype'],
                                             all_atom_positions=pdb['batch']["all_atom_positions"],
                                             all_atom_mask=pdb['batch']["all_atom_mask"])

    dm = np.sqrt(np.square(x_beta[:,None] - x_beta[None,:]).sum(-1))
    dm_mask = np.where(dm < 22, 1, 0)
    
    # mask_dist
    mask = np.zeros([self._len, self._len], dtype=np.float32)
    for iclique in cliques:
      for ichain in iclique:
        istart, istop = boundaries[ichain], boundaries[ichain+1]
        for jchain in iclique:
          jstart, jstop = boundaries[jchain], boundaries[jchain+1]
          mask[istart:istop, jstart:jstop] = 1
    mask += mask.T
    mask = np.where(mask == 0, 0., 1.)
    self._batch['mask_d'] = jnp.array(mask * dm_mask)
    self._batch['mask_dgram'] = mask

    # mask chain weights
    if chain_weights is not None:
      for ik, iv in chain_weights.items():
        if ik in chains:
          ind = chains.index(ik)
          istart, istop = boundaries[ind], boundaries[ind+1]
          mask[istart:istop, istart:istop] *= iv

    self._batch['mask_d_w'] = mask * dm_mask

    # update template features
    if self._args["use_templates"]:
      self._update_template(self._inputs, self.opt, self.key())

    # template_mask = np.zeros([num_templates, self._len, self._len])
    # if split_templates:
    #   for ith, iclique in enumerate(self.cliques):
    #     i_mask = np.zeros([self._len])
    #     for ichain in iclique:
    #       i_mask[boundaries[ichain]:boundaries[ichain+1]] = 1
    #     template_mask[ith] = i_mask[:, None] * i_mask[None, :]
    #   # make sure all the attentions can focus on at least one pixel
    #   remain_mask = 1. - template_mask.sum(0)
    #   template_mask[0] += remain_mask
    # else:
    #   template_mask[0] = 1
    template_mask = np.ones(num_templates)
    template_mask[-1] = 0
    
    self._inputs["template_mask"] = jnp.array(template_mask)
    self._inputs["mask_template_interchain"] = False
    self._inputs["use_dropout"] = False

    # update residue index
    if len(self.lens) > 1:
      boundaries = [0] + list(np.cumsum(self.lens))
      residue_index = np.array(self._inputs['residue_index'])
      for ith in range(len(boundaries)-1):
        residue_index[boundaries[ith]:boundaries[ith+1]] += 50*ith
      if not self._args['use_multimer']:
        self._inputs['residue_index'] = jnp.array(residue_index)
      self.residue_index = residue_index
    else:
      raise Exception('current is only suitable for complex!')
    
    # update asym_id, sym_id, entity_id
    if self._args['use_multimer']:
      asym_id = np.zeros(self._len)
      for ith in range(len(boundaries)-1):
        start, stop = boundaries[ith], boundaries[ith+1]
        asym_id[start:stop] = ith
      
      entity_id = np.zeros(self._len)
      sym_id = np.zeros(self._len)
      unique_seqs = list(set(seqs))
      for ith, iseq in enumerate(unique_seqs):
        idx = [jth for jth, jseq in enumerate(seqs) if jseq==iseq]
        for jth, ind in enumerate(idx):
          start, stop = boundaries[ind], boundaries[ind+1]
          sym_id[start:stop] = jth
          entity_id[start:stop] = ith
      self._inputs['asym_id'] = jnp.array(asym_id)
      self._inputs['entity_id'] = jnp.array(entity_id)
      self._inputs['sym_id'] = jnp.array(sym_id)

    # update amino acid sidechain identity
    update_aatype(self._wt_aatype, self._inputs)

    # make config for msa generation
    if use_multimer:
      cfg_common = self._runner.config.model
      cfg_common.num_recycle = 0
    else:
      cfg_common = self._runner.config.data.common
      cfg_common.max_extra_msa = 0
      cfg_common.num_recycle = 0
      self._runner.config.data.eval.max_msa_clusters = num_msa

    if self._args["use_msa"]:
      feature_msa = {**pipeline.make_sequence_features(sequence=wt_seq, description="none", num_res=self._len),
                     **pipeline.make_msa_features(msas=[input_msa], deletion_matrices=[input_dlm])}
      self.feature_msa = self._runner.process_features(feature_msa, random_seed=0)
      self.feature_msa = jax.tree_map(lambda x:jnp.array(x), self.feature_msa)

    self._opt = copy_dict(self.opt)
    self.restart(**kwargs)
    
#######################
# utils
#######################
def repeat_idx(idx, copies=1, offset=50):
  idx_offset = np.repeat(np.cumsum([0]+[idx[-1]+offset]*(copies-1)),len(idx))
  return np.tile(idx,copies) + idx_offset

def prep_pdb(pdb_filename, chain=None, for_alphafold=True):
  '''extract features from pdb'''

  def add_cb(batch):
    '''add missing CB atoms based on N,CA,C'''
    p,m = batch["all_atom_positions"],batch["all_atom_mask"]
    atom_idx = residue_constants.atom_order
    atoms = {k:p[...,atom_idx[k],:] for k in ["N","CA","C"]}
    cb = atom_idx["CB"]
    cb_atoms = _np_get_cb(**atoms, use_jax=False)
    cb_mask = np.prod([m[...,atom_idx[k]] for k in ["N","CA","C"]],0)
    batch["all_atom_positions"][...,cb,:] = np.where(m[:,cb,None], p[:,cb,:], cb_atoms)
    batch["all_atom_mask"][...,cb] = (m[:,cb] + cb_mask) > 0

  # go through each defined chain
  chains = [None] if chain is None else chain.split(",")
  o,last = [],0
  residue_idx, chain_idx = [],[]
  for chain in chains:
    protein_obj = protein.from_pdb_string(pdb_to_string(pdb_filename), chain_id=chain)
    batch = {'aatype': protein_obj.aatype,
             'all_atom_positions': protein_obj.atom_positions,
             'all_atom_mask': protein_obj.atom_mask}

    add_cb(batch) # add in missing cb (in the case of glycine)

    has_ca = batch["all_atom_mask"][:,0] == 1
    batch = jax.tree_map(lambda x:x[has_ca], batch)
    seq = "".join([order_aa[a] for a in batch["aatype"]])
    residue_index = protein_obj.residue_index[has_ca] + last      
    last = residue_index[-1] + 50
    
    if for_alphafold:
      batch.update(all_atom.atom37_to_frames(**batch))
      template_aatype = residue_constants.sequence_to_onehot(seq, residue_constants.HHBLITS_AA_TO_ID)
      template_features = {"template_aatype":template_aatype,
                           "template_all_atom_masks":batch["all_atom_mask"],
                           "template_all_atom_positions":batch["all_atom_positions"]}
      o.append({"batch":batch,
                "template_features":template_features,
                "residue_index": residue_index})
    else:        
      o.append({"batch":batch,
                "residue_index": residue_index})
    
    residue_idx.append(protein_obj.residue_index[has_ca])
    chain_idx.append([chain] * len(residue_idx[-1]))

  # concatenate chains
  o = jax.tree_util.tree_map(lambda *x:np.concatenate(x,0),*o)
  
  if for_alphafold:
    o["template_features"] = jax.tree_map(lambda x:x[None],o["template_features"])
    o["template_features"]["template_domain_names"] = np.asarray(["None"])

  # save original residue and chain index
  o["idx"] = {"residue":np.concatenate(residue_idx), "chain":np.concatenate(chain_idx)}
  return o

def make_fixed_size(feat, model_runner, length, batch_axis=True):
  '''pad input features'''
  cfg = model_runner.config
  if batch_axis:
    shape_schema = {k:[None]+v for k,v in dict(cfg.data.eval.feat).items()}
  else:
    shape_schema = {k:v for k,v in dict(cfg.data.eval.feat).items()}
  num_msa_seq = cfg.data.eval.max_msa_clusters - cfg.data.eval.max_templates
  pad_size_map = {
      shape_placeholders.NUM_RES: length,
      shape_placeholders.NUM_MSA_SEQ: num_msa_seq,
      shape_placeholders.NUM_EXTRA_SEQ: cfg.data.common.max_extra_msa,
      shape_placeholders.NUM_TEMPLATES: cfg.data.eval.max_templates
  }
  for k, v in feat.items():
    # Don't transfer this to the accelerator.
    if k == 'extra_cluster_assignment':
      continue
    shape = list(v.shape)
    schema = shape_schema[k]
    assert len(shape) == len(schema), (
        f'Rank mismatch between shape and shape schema for {k}: '
        f'{shape} vs {schema}')
    pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
    padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
    feat[k] = np.pad(v, padding)
  return feat

def get_sc_pos(aa_ident, atoms_to_exclude=None):
  '''get sidechain indices/weights for all_atom14_positions'''

  # decide what atoms to exclude for each residue type
  a2e = {}
  for r in resname_to_idx:
    if isinstance(atoms_to_exclude,dict):
      a2e[r] = atoms_to_exclude.get(r,atoms_to_exclude.get("ALL",["N","C","O"]))
    else:
      a2e[r] = ["N","C","O"] if atoms_to_exclude is None else atoms_to_exclude

  # collect atom indices
  pos,pos_alt = [],[]
  N,N_non_amb = [],[]
  for n,a in enumerate(aa_ident):
    aa = idx_to_resname[a]
    atoms = set(residue_constants.residue_atoms[aa])
    atoms14 = residue_constants.restype_name_to_atom14_names[aa]
    swaps = residue_constants.residue_atom_renaming_swaps.get(aa,{})
    swaps.update({v:k for k,v in swaps.items()})
    for atom in atoms.difference(a2e[aa]):
      pos.append(n * 14 + atoms14.index(atom))
      if atom in swaps:
        pos_alt.append(n * 14 + atoms14.index(swaps[atom]))
      else:
        pos_alt.append(pos[-1])
        N_non_amb.append(n)
      N.append(n)

  pos, pos_alt = np.asarray(pos), np.asarray(pos_alt)
  non_amb = pos == pos_alt
  N, N_non_amb = np.asarray(N), np.asarray(N_non_amb)
  w = np.array([1/(n == N).sum() for n in N])
  w_na = np.array([1/(n == N_non_amb).sum() for n in N_non_amb])
  w, w_na = w/w.sum(), w_na/w_na.sum()
  return {"pos":pos, "pos_alt":pos_alt, "non_amb":non_amb,
          "weight":w, "weight_non_amb":w_na[:,None]}

def prep_input_features(L, N=1, T=1, eN=1):
  '''
  given [L]ength, [N]umber of sequences and number of [T]emplates
  return dictionary of blank features
  '''
  inputs = {'aatype': np.zeros(L,int),
            'target_feat': np.zeros((L,20)),
            'msa_feat': np.zeros((N,L,49)),
            # 23 = one_hot -> (20, UNK, GAP, MASK)
            # 1  = has deletion
            # 1  = deletion_value
            # 23 = profile
            # 1  = deletion_mean_value
  
            'seq_mask': np.ones(L),
            'msa_mask': np.ones((N,L)),
            'msa_row_mask': np.ones(N),
            'atom14_atom_exists': np.zeros((L,14)),
            'atom37_atom_exists': np.zeros((L,37)),
            'residx_atom14_to_atom37': np.zeros((L,14),int),
            'residx_atom37_to_atom14': np.zeros((L,37),int),            
            'residue_index': np.arange(L),
            'extra_deletion_value': np.zeros((eN,L)),
            'extra_has_deletion': np.zeros((eN,L)),
            'extra_msa': np.zeros((eN,L),int),
            'extra_msa_mask': np.zeros((eN,L)),
            'extra_msa_row_mask': np.zeros(eN),

            # for template inputs
            'template_aatype': np.zeros((T,L),int),
            'template_all_atom_mask': np.zeros((T,L,37)),
            'template_all_atom_positions': np.zeros((T,L,37,3)),
            'template_mask': np.ones(T),
            'template_pseudo_beta': np.zeros((T,L,3)),
            'template_pseudo_beta_mask': np.zeros((T,L)),

            # for alphafold-multimer
            'asym_id': np.zeros(L),
            'sym_id': np.zeros(L),
            'entity_id': np.zeros(L),
            'all_atom_positions': np.zeros((N,37,3))}
  return inputs

def cal_seqid(alignment):
  aliA = alignment.seqA
  seqA = ''.join([iaa for iaa in aliA if iaa != '-'])
  aliB = alignment.seqB
  seqB = ''.join([iaa for iaa in aliB if iaa != '-'])
  idx = [ind for ind in range(len(aliA)) if aliA[ind] != '-']
  aliA = ''.join([aliA[ind] for ind in idx])
  aliB = ''.join([aliB[ind] for ind in idx])
  seqid = sum([1 for ith in range(len(aliA)) if aliA[ith] == aliB[ith]])
  return seqid / min(len(seqA), len(seqB))
