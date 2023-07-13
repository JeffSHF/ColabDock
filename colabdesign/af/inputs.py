import jax
import jax.numpy as jnp
import numpy as np

from colabdesign.shared.model import soft_seq
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.alphafold.model import model

############################################################################
# AF_INPUTS - functions for modifying inputs before passing to alphafold
############################################################################
class _af_inputs:
  def _get_seq(self, params, opt, aux, key):
    '''get sequence features'''
    seq = soft_seq(params["seq"], opt, key)
    if "pos" in opt and "fix_seq" in opt:
      seq_ref = jax.nn.one_hot(self._wt_aatype,20)
      p = opt["pos"]
      if self.protocol == "partial":
        fix_seq = lambda x:jnp.where(opt["fix_seq"],x.at[...,p,:].set(seq_ref),x)
      else:
        fix_seq = lambda x:jnp.where(opt["fix_seq"],x.at[...,p,:].set(seq_ref[...,p,:]),x)
      seq = jax.tree_map(fix_seq,seq)
    aux.update({"seq":seq, "seq_pseudo":seq["pseudo"]})
    
    # protocol specific modifications to seq features
    if self.protocol == "binder":
      # concatenate target and binder sequence
      seq_target = jax.nn.one_hot(self._batch["aatype"][:self._target_len],20)
      seq_target = jnp.broadcast_to(seq_target,(self._num, *seq_target.shape))
      seq = jax.tree_map(lambda x:jnp.concatenate([seq_target,x],1), seq)
      
    if self.protocol in ["fixbb","hallucination"] and self._copies > 1:
      seq = jax.tree_map(lambda x:expand_copies(x, self._copies, self._args["block_diag"]), seq)
    return seq


  def _update_template(self, inputs, opt, key):
    ''''dynamically update template features'''

    # aatype = is used to define template's CB coordinates (CA in case of glycine)
    # template_aatype = is used as template's sequence

    L = self._len
    if self._args["rm_template_seq"]:
      aatype = self._batch["aatype"]
      template_aatype = jnp.broadcast_to(opt["template"]["aatype"],(L,))
    else:
      template_aatype = aatype = self._batch["aatype"]
      
    # get pseudo-carbon-beta coordinates (carbon-alpha for glycine)
    pb, pb_mask = model.modules.pseudo_beta_fn(aatype,
                                               self._batch["all_atom_positions"],
                                               self._batch["all_atom_mask"])
      
    # define template features
    template_feats = {"template_aatype": template_aatype,
                      "template_all_atom_positions": self._batch["all_atom_positions"],
                      "template_all_atom_mask": self._batch["all_atom_mask"],
                      "template_pseudo_beta": pb,
                      "template_pseudo_beta_mask": pb_mask}

    # protocol specific template injection
    bounds = [0] + list(np.cumsum(self.lens))
    for k, v in template_feats.items():
      if self.split_templates:
        for ith, iclique in enumerate(self.cliques):
          for ichain in iclique:
            v_set = v[bounds[ichain]:bounds[ichain + 1]]
            inputs[k] = inputs[k].at[ith, bounds[ichain]:bounds[ichain + 1]].set(v_set)
      else:
        inputs[k] = inputs[k].at[0].set(v)
        
      if k == "template_all_atom_masks" and self._args["rm_template_seq"]:
        inputs[k] = inputs[k].at[:,:,5:].set(0)

    # dropout template input features
    L = inputs["template_aatype"].shape[1]
    n = self._target_len if self.protocol == "binder" else 0
    pos_mask = jax.random.bernoulli(key, 1-opt["template"]["dropout"],(L,))
    inputs["template_all_atom_mask"] = inputs["template_all_atom_mask"].at[:,n:].multiply(pos_mask[n:,None])
    inputs["template_pseudo_beta_mask"] = inputs["template_pseudo_beta_mask"].at[:,n:].multiply(pos_mask[n:])
  
  def _update_template_complex(self, inputs, atom_position, atom_mask):
    '''update the generated complex structure as the last template'''
    pb, pb_mask = model.modules.pseudo_beta_fn(self._wt_aatype,
                                               atom_position,
                                               atom_mask)
    inputs["template_all_atom_positions"] = inputs["template_all_atom_positions"].at[-1].set(atom_position)
    inputs["template_all_atom_mask"] = inputs["template_all_atom_mask"].at[-1].set(atom_mask)
    inputs["template_pseudo_beta"] = inputs["template_pseudo_beta"].at[-1].set(pb)
    inputs["template_pseudo_beta_mask"] = inputs["template_pseudo_beta_mask"].at[-1].set(pb_mask)
    if not self._args["rm_template_seq"]:
      inputs["template_all_atom_mask"] = inputs["template_all_atom_mask"].at[:,:,5:].set(0)
    else:
      inputs["template_aatype"] = inputs["template_aatype"].at[-1].set(self._wt_aatype)


def update_seq(seq, inputs, seq_1hot=None, seq_pssm=None, msa_input=None):
  '''update the sequence features'''
  
  if seq_1hot is None: seq_1hot = seq 
  if seq_pssm is None: seq_pssm = seq
  
  seq_1hot = jnp.pad(seq_1hot,[[0,0],[0,0],[0,22-seq_1hot.shape[-1]]])
  seq_pssm = jnp.pad(seq_pssm,[[0,0],[0,0],[0,22-seq_pssm.shape[-1]]])
  
  if msa_input is None:
    msa_feat = jnp.zeros_like(inputs["msa_feat"]).at[0, :, :22].set(seq_1hot[0]).at[0, :, 25:47].set(seq_pssm[0])
  else:
    msa_feat = jnp.array(msa_input['msa_feat'][0]).at[0, :, :22].set(seq_1hot[0])
  if seq.ndim == 3:
    target_feat = jnp.zeros_like(inputs["target_feat"]).at[...,:20].set(seq[0,...,:20])
  else:
    target_feat = jnp.zeros_like(inputs["target_feat"]).at[...,:20].set(seq[...,:20])
    
  inputs.update({"target_feat":target_feat,"msa_feat":msa_feat})

def update_aatype(aatype, inputs):
  r = residue_constants
  a = {"atom14_atom_exists":r.restype_atom14_mask,
       "atom37_atom_exists":r.restype_atom37_mask,
       "residx_atom14_to_atom37":r.restype_atom14_to_atom37,
       "residx_atom37_to_atom14":r.restype_atom37_to_atom14}
  mask = inputs["seq_mask"][:,None]
  inputs.update(jax.tree_map(lambda x:jnp.where(mask,jnp.asarray(x)[aatype],0), a))
  inputs["aatype"] = aatype

def expand_copies(x, copies, block_diag=True):
  '''
  given msa (N,L,20) expand to (N*(1+copies),L*copies,22) if block_diag else (N,L*copies,22)
  '''
  if x.shape[-1] < 22:
    x = jnp.pad(x,[[0,0],[0,0],[0,22-x.shape[-1]]])
  x = jnp.tile(x,[1,copies,1])
  if copies > 1 and block_diag:
    L = x.shape[1]
    sub_L = L // copies
    y = x.reshape((-1,1,copies,sub_L,22))
    block_diag_mask = jnp.expand_dims(jnp.eye(copies),(0,3,4))
    seq = block_diag_mask * y
    gap_seq = (1-block_diag_mask) * jax.nn.one_hot(jnp.repeat(21,sub_L),22)  
    y = (seq + gap_seq).swapaxes(0,1).reshape(-1,L,22)
    return jnp.concatenate([x,y],0)
  else:
    return x

def crop_feat(feat, pos, model_runner, add_batch=True):  
  def find(x,k):
    i = []
    for j,y in enumerate(x):
      if y == k: i.append(j)
    return i
  
  if feat is None:
    return None

  else:
    cfg = model_runner.config
    shapes = cfg.data.eval.feat
    NUM_RES = "num residues placeholder"
    idx = {k:find(v,NUM_RES) for k,v in shapes.items()}

    new_feat = {}
    for k,v in feat.items():
      # v_ = v.copy()
      if k in idx and len(idx[k]) != 0:
        for i in idx[k]:
          v_ = jnp.take(v, pos, i + add_batch)
        new_feat[k] = v_
      elif k == 'prev':
        # set prev manually
        new_feat['prev'] = {}
        new_feat['prev']['prev_msa_first_row'] = v['prev_msa_first_row'][pos]
        new_feat['prev']['prev_pair'] = v['prev_pair'][pos, :][:, pos]
        new_feat['prev']['prev_pos'] = v['prev_pos'][pos]
      elif k == 'mask_d':
        new_feat[k] = v[pos, :][:, pos]
      elif k == 'masks_rest':
        masks = {}
        for ik, iv in v.items():
          if iv is not None:
            if ik == 'MvN':
              mat = iv[:, pos, :][:, :, pos]
            else:
              mat = iv[pos, :][:, pos]
            masks[ik] = mat
          else:
            masks[ik] = None
        new_feat['masks_rest'] = masks
      else:
        new_feat[k] = v.copy()
    return new_feat