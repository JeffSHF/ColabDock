import jax
import jax.numpy as jnp

from colabdesign.shared.protein import _np_kabsch, _np_get_6D_loss
from colabdesign.af.alphafold.model import model, folding
from colabdesign.af.alphafold.common import confidence_jax

####################################################
# AF_LOSS - setup loss function
####################################################
class _af_loss:
  def _loss_dock(self, inputs, outputs, opt, aux, batch=None):
    '''get losses'''
    if batch is None: batch = self._batch
    copies = self._copies

    # plddt
    plddt_prob = jax.nn.softmax(outputs["predicted_lddt"]["logits"])
    plddt_loss = (plddt_prob * jnp.arange(plddt_prob.shape[-1])[::-1]).mean(-1)
    aux["losses"].update({"plddt": plddt_loss.mean()})

    # PAE
    self._get_dock_pw_loss(inputs, outputs, opt, aux, interface=True,
                           mask_d=batch['mask_d'])

    # dgram loss
    aatype = inputs["aatype"]
    dgram_cce = get_dgram_loss(batch, outputs, copies=copies, aatype=aatype, return_cce=True)
    dgram_cce = (dgram_cce * batch['mask_d_w']).sum() / (batch['mask_d'].sum() + 1e-8)
    aux["losses"].update({"dgram_cce": dgram_cce})

    dgram = outputs["distogram"]["logits"]
    dgram_bins = jnp.append(0, jnp.linspace(2.3125, 21.6875, dgram.shape[-1]-1))
    x = get_pw_con_loss(dgram, dgram_bins,
                        cutoff=opt["rest"]["cutoff"],
                        binary=opt["rest"]["binary"],
                        entropy=opt["rest"]["entropy"])
    
    # restraints loss
    mask_1v1 = batch['masks_rest']["1v1"]
    if mask_1v1 is not None:
      results = _get_rest1v1_loss(x=x, mask=mask_1v1)
      aux["losses"].update({"rest_1v1": results})
    
    mask_1vN = batch['masks_rest']["1vN"]
    if mask_1vN is not None:
      num_r = opt["rest"]["num"]
      results = _get_rest1vN_loss(x=x, mask=mask_1vN, num_r=num_r)
      aux["losses"].update({"rest_1vN": results})
    
    mask_MvN = batch['masks_rest']["MvN"]
    if mask_MvN is not None:
      num_r = opt["rest"]["num"]
      num_cs = opt["num_MvN"]
      results = _get_restMvN_loss(x=x, mask=mask_MvN, num_r=num_r, num_cs=num_cs)
      aux["losses"].update({"rest_MvN": results})
    
    mask_non = batch['masks_rest']["non"]
    if mask_non is not None:
      results = _get_restnon_loss(outputs=outputs, opt=opt, mask=mask_non)
      aux["losses"].update({"rest_non": results})

  def _get_dock_pw_loss(self, inputs, outputs, opt, aux, interface=False, mask_d=None):
    '''get pairwise loss features'''

    # decide on what offset to use
    if "offset" in inputs:
      offset = inputs["offset"]
    else:
      idx = inputs["residue_index"]
      offset = idx[:,None] - idx[None,:]

    # pae loss
    pae_prob = jax.nn.softmax(outputs["predicted_aligned_error"]["logits"])
    pae = (pae_prob * jnp.arange(pae_prob.shape[-1])).mean(-1)
    
    # define distogram
    dgram = outputs["distogram"]["logits"]
    dgram_bins = jnp.append(0, outputs["distogram"]["bin_edges"])
    if not interface:
      aux["losses"].update({"con":get_con_loss(dgram, dgram_bins, offset=offset, **opt["con"]).mean(),
                            "pae":pae.mean()})
    else:
      # split pae/con into inter/intra
      for k,v in zip(["pae","con"], [pae,dgram]):
        if k == "con":
          x = get_con_loss(v, dgram_bins, offset=offset,
                           mask_intra=mask_d, **opt["con"]).mean()
          ix = get_con_loss(v, dgram_bins, offset=offset,
                            mask_intra=1-mask_d, **opt["i_con"]).mean()
        else:
          x = (v * mask_d).sum() / (mask_d.sum() + 1e-8)
          ix = (v * (1-mask_d)).sum() / ((1-mask_d).sum() + 1e-8)
        aux["losses"].update({f"{k}":x,f"i_{k}":ix})

#####################################################################################
#####################################################################################
def _get_rest1v1_loss(x, mask):
  '''
  get loss on 1v1 restrains
  '''
  return (x * mask).sum() / (mask.sum() + 1e-8)

def _get_rest1vN_loss(x, mask, num_r, num_c=None):
  '''
  get loss on 1vN restrains
  '''
  mask_1d = mask.sum(-1)
  mask_1d = jnp.where(mask_1d, 1.0, 0.0)
  if num_c is None: num_c = mask_1d.sum()

  maxi = jax.lax.stop_gradient(x.max())
  x = x * mask +  (1.0 - mask) * (maxi + 100)
  x = jnp.sort(x, -1)
  x = jnp.sort(x, 0)

  a,b = x.shape
  r_mask = jnp.arange(b) < num_r
  r_mask = jnp.repeat(r_mask[None], repeats=a, axis=0)
  c_mask = jnp.arange(a) < num_c
  c_mask = jnp.repeat(c_mask[:, None], repeats=b, axis=1)
  k_mask = r_mask * c_mask

  results = jnp.where(k_mask, x, 0.0).sum() / (k_mask.sum() + 1e-8)
  return results

def _get_restMvN_loss(x, mask, num_r, num_cs):
  '''
  get loss on MvN restrains
  '''
  results = []
  for i in range(len(mask)):
    imask = mask[i]
    inum_c = num_cs[i]
    results.append(_get_rest1vN_loss(x, imask, num_r=num_r, num_c=inum_c))
  results = jnp.array(results)
  return results.mean()

def _get_restnon_loss(outputs, opt, mask):
  '''
  get loss on restrains
  '''
  dgram = outputs["distogram"]["logits"]
  dgram_bins = jnp.append(0,jnp.linspace(2.3125, 21.6875, dgram.shape[-1] - 1))

  def get_pw_noncon_loss(dgram, cutoff, binary=True, entropy=True):
    '''convert distogram into pairwise contact loss'''
    bins = dgram_bins > cutoff

    px = jax.nn.softmax(dgram)
    px_ = jax.nn.softmax(dgram - 1e7 * (1-bins))

    con_loss_cat = 1 - (bins * px).max(-1)
    con_loss_bin = 1 - (bins * px).sum(-1)
    con_loss = jnp.where(binary, con_loss_bin, con_loss_cat)

    # binary/cateogorical cross-entropy
    con_loss_cat_ent = -(px_ * jax.nn.log_softmax(dgram)).sum(-1)
    con_loss_bin_ent = -jnp.log((bins * px).sum(-1) + 1e-8)
    
    con_loss_ent = jnp.where(binary, con_loss_bin_ent, con_loss_cat_ent)
    return jnp.where(entropy, con_loss_ent, con_loss)

  x_non = get_pw_noncon_loss(dgram, 
                             cutoff=opt["rest_non"]["cutoff"],
                             binary=opt["rest_non"]["binary"],
                             entropy=opt["rest_non"]["entropy"])

  return (x_non * mask).sum() / (mask.sum() + 1e-8)


def get_pw_con_loss(dgram, dgram_bins, cutoff, binary=True, entropy=True):
  '''convert distogram into pairwise contact loss'''
  bins = dgram_bins < cutoff
  
  px = jax.nn.softmax(dgram)
  px_ = jax.nn.softmax(dgram - 1e7 * (1-bins))
  
  con_loss_cat = 1 - (bins * px).max(-1)
  con_loss_bin = 1 - (bins * px).sum(-1)
  con_loss = jnp.where(binary, con_loss_bin, con_loss_cat)
  
  # binary/cateogorical cross-entropy
  con_loss_cat_ent = -(px_ * jax.nn.log_softmax(dgram)).sum(-1)
  con_loss_bin_ent = -jnp.log((bins * px).sum(-1) + 1e-8)
  
  con_loss_ent = jnp.where(binary, con_loss_bin_ent, con_loss_cat_ent)
  return jnp.where(entropy, con_loss_ent, con_loss)


def get_con_loss(dgram, dgram_bins, cutoff=None, binary=True,
                 num=1, seqsep=0, offset=None, mask_intra=None):
  '''convert distogram into contact loss'''  
  if cutoff is None: cutoff = dgram_bins[-1]
  x = get_pw_con_loss(dgram, dgram_bins, cutoff, binary, entropy=True)  
  a,b = x.shape
  if offset is None:
    mask = jnp.abs(jnp.arange(a)[:,None] - jnp.arange(b)[None,:]) >= seqsep
  else:
    mask = jnp.abs(offset) >= seqsep
  
  if mask_intra is not None:
    mask = mask * mask_intra
  x = jnp.sort(jnp.where(mask, x, jnp.nan))
  k_mask = (jnp.arange(b) < num) * (jnp.isnan(x) == False)    
  return jnp.where(k_mask, x, 0.0).sum(-1) / (k_mask.sum(-1) + 1e-8)

####################
# confidence metrics
####################
def get_plddt(outputs):
  logits = outputs["predicted_lddt"]["logits"]
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = jnp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.sum(probs * bin_centers[None, :], axis=-1)

def get_pae(outputs):
  prob = jax.nn.softmax(outputs["predicted_aligned_error"]["logits"],-1)
  breaks = outputs["predicted_aligned_error"]["breaks"]
  step = breaks[1]-breaks[0]
  bin_centers = breaks + step/2
  bin_centers = jnp.append(bin_centers,bin_centers[-1]+step)
  return (prob*bin_centers).sum(-1)

def get_ptm(outputs):
  dic = {'logits': outputs['predicted_aligned_error']['logits'], 'breaks': outputs['predicted_aligned_error']['breaks']}
  ptm = confidence_jax.predicted_tm_score_jax(outputs['predicted_aligned_error']['logits'],
                                              outputs['predicted_aligned_error']['breaks'])
  dic['value'] = ptm
  return dic


####################
# loss functions
####################
def get_dgram_loss(batch, outputs=None, copies=1, aatype=None, pred=None, return_cce=False):

  # gather features
  if aatype is None: aatype = batch["aatype"]
  if pred is None: pred = outputs["distogram"]["logits"]

  # get true features
  x, weights = model.modules.pseudo_beta_fn(aatype=aatype,
                                            all_atom_positions=batch["all_atom_positions"],
                                            all_atom_mask=batch["all_atom_mask"])

  dm = jnp.square(x[:,None]-x[None,:]).sum(-1,keepdims=True)
  bin_edges = jnp.linspace(2.3125, 21.6875, pred.shape[-1] - 1)
  true = jax.nn.one_hot((dm > jnp.square(bin_edges)).sum(-1), pred.shape[-1])

  return _get_dgram_loss(true, pred, weights, copies, return_cce=return_cce)


def _get_dgram_loss(true, pred, weights=None, copies=1, return_cce=False):
  
  length = true.shape[0]
  if weights is None: weights = jnp.ones(length)
  F = {"t":true, "p":pred, "m":weights[:,None] * weights[None,:]}  

  def cce_fn(t,p,m):
    cce = -(t*jax.nn.log_softmax(p)).sum(-1)
    return cce, (cce*m).sum((-1,-2))/(m.sum((-1,-2))+1e-8)

  if copies > 1:
    (L,C) = (length//copies, copies-1)

    # intra (L,L,F)
    intra = jax.tree_map(lambda x:x[:L,:L], F)
    cce, cce_loss = cce_fn(**intra)

    # inter (C*L,L,F)
    inter = jax.tree_map(lambda x:x[L:,:L], F)
    if C == 0:
      i_cce, i_cce_loss = cce_fn(**inter)

    else:
      # (C,L,L,F)
      inter = jax.tree_map(lambda x:x.reshape(C,L,L,-1), inter)
      inter = {"t":inter["t"][:,None],        # (C,1,L,L,F)
               "p":inter["p"][None,:],        # (1,C,L,L,F)
               "m":inter["m"][:,None,:,:,0]}  # (C,1,L,L)             
      
      # (C,C,L,L,F) → (C,C,L,L) → (C,C) → (C) → ()
      i_cce, i_cce_loss = cce_fn(**inter)
      i_cce_loss = sum([i_cce_loss.min(i).sum() for i in [0,1]]) / 2

    total_loss = (cce_loss + i_cce_loss) / copies
    return (cce, i_cce) if return_cce else total_loss

  else:
    cce, cce_loss = cce_fn(**F)
    return cce if return_cce else cce_loss

def get_rmsd_loss(batch, outputs, L=None, include_L=True, copies=1):
  true = batch["all_atom_positions"][:,1,:]
  pred = outputs["structure_module"]["final_atom_positions"][:,1,:]
  weights = batch["all_atom_mask"][:,1]
  return _get_rmsd_loss(true, pred, weights=weights, L=L, include_L=include_L, copies=copies)

def _get_rmsd_loss(true, pred, weights=None, L=None, include_L=True, copies=1):
  '''
  get rmsd + alignment function
  align based on the first L positions, computed weighted rmsd using all 
  positions (if include_L=True) or remaining positions (if include_L=False).
  '''
  # normalize weights
  length = true.shape[-2]
  if weights is None:
    weights = (jnp.ones(length)/length)[...,None]
  else:
    weights = (weights/(weights.sum(-1,keepdims=True) + 1e-8))[...,None]

  # determine alignment [L]ength and remaining [l]ength
  if copies > 1:
    if L is None:
      L = iL = length // copies; C = copies-1
    else:
      (iL,C) = ((length-L) // copies, copies)
  else:
    (L,iL,C) = (length,0,0) if L is None else (L,length-L,1)

  # slice inputs
  if iL == 0:
    (T,P,W) = (true,pred,weights)
  else:
    (T,P,W) = (x[...,:L,:] for x in (true,pred,weights))
    (iT,iP,iW) = (x[...,L:,:] for x in (true,pred,weights))

  # get alignment and rmsd functions
  (T_mu,P_mu) = ((x*W).sum(-2,keepdims=True)/W.sum((-1,-2)) for x in (T,P))
  aln = _np_kabsch((P-P_mu)*W, T-T_mu)   
  align_fn = lambda x: (x - P_mu) @ aln + T_mu
  msd_fn = lambda t,p,w: (w*jnp.square(align_fn(p)-t)).sum((-1,-2))
  
  # compute rmsd
  if iL == 0:
    msd = msd_fn(true,pred,weights)
  elif C > 1:
    # all vs all alignment of remaining, get min RMSD
    iT = iT.reshape(-1,C,1,iL,3).swapaxes(0,-3)
    iP = iP.reshape(-1,1,C,iL,3).swapaxes(0,-3)
    imsd = msd_fn(iT, iP, iW.reshape(-1,C,1,iL,1).swapaxes(0,-3))
    imsd = (imsd.min(0).sum(0) + imsd.min(1).sum(0)) / 2 
    imsd = imsd.reshape(jnp.broadcast_shapes(true.shape[:-2],pred.shape[:-2]))
    msd = (imsd + msd_fn(T,P,W)) if include_L else (imsd/iW.sum((-1,-2)))
  else:
    msd = msd_fn(true,pred,weights) if include_L else (msd_fn(iT,iP,iW)/iW.sum((-1,-2)))
  rmsd = jnp.sqrt(msd + 1e-8)

  return {"rmsd":rmsd, "align":align_fn}

def get_sc_rmsd(true, pred, sc):
  '''get sidechain rmsd + alignment function'''

  # select atoms
  (T, P) = (true.reshape(-1,3), pred.reshape(-1,3))
  (T, T_alt, P) = (T[sc["pos"]], T[sc["pos_alt"]], P[sc["pos"]])

  # select non-ambigious atoms
  (T_na, P_na) = (T[sc["non_amb"]], P[sc["non_amb"]])

  # get alignment of non-ambigious atoms
  if "weight_non_amb" in sc:
    T_mu_na = (T_na * sc["weight_non_amb"]).sum(0)
    P_mu_na = (P_na * sc["weight_non_amb"]).sum(0)
    aln = _np_kabsch((P_na-P_mu_na) * sc["weight_non_amb"], T_na-T_mu_na)
  else:
    T_mu_na, P_mu_na = T_na.mean(0), P_na.mean(0)
    aln = _np_kabsch(P_na-P_mu_na, T_na-T_mu_na)

  # apply alignment to all atoms
  align_fn = lambda x: (x - P_mu_na) @ aln + T_mu_na
  P = align_fn(P)

  # compute rmsd
  sd = jnp.minimum(jnp.square(P-T).sum(-1), jnp.square(P-T_alt).sum(-1))
  if "weight" in sc:
    msd = (sd*sc["weight"]).sum()
  else:
    msd = sd.mean()
  rmsd = jnp.sqrt(msd + 1e-8)
  return {"rmsd":rmsd, "align":align_fn}


#--------------------------------------
# TODO (make copies friendly)
#--------------------------------------
def get_fape_loss(batch, outputs, model_config, use_clamped_fape=False):
  sub_batch = jax.tree_map(lambda x: x, batch)
  sub_batch["use_clamped_fape"] = use_clamped_fape
  loss = {"loss":0.0}    
  folding.backbone_loss(loss, sub_batch, outputs["structure_module"], model_config.model.heads.structure_module)
  return loss["loss"]

def get_6D_loss(batch, outputs, **kwargs):
  true = batch["all_atom_positions"]
  pred = outputs["structure_module"]["final_atom_positions"]
  mask = batch["all_atom_mask"]
  return _np_get_6D_loss(true, pred, mask, **kwargs)
