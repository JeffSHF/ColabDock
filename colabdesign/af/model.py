import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import joblib

from colabdesign.af.alphafold.model import data, config, model

from colabdesign.shared.model import design_model
from colabdesign.shared.utils import Key

from colabdesign.af.prep   import _af_prep
from colabdesign.af.loss   import _af_loss, get_plddt, get_pae, get_ptm
from colabdesign.af.utils  import _af_utils, save_pdb_from_aux, cal_iptm
from colabdesign.af.design import _af_design
from colabdesign.af.inputs import _af_inputs, update_seq, crop_feat
################################################################
# MK_DESIGN_MODEL - initialize model, and put it all together
################################################################

class mk_af_model(design_model, _af_inputs, _af_loss, _af_prep, _af_design, _af_utils):
  def __init__(self, protocol="dock", num_seq=1,
               num_models=1, sample_models=True,
               recycle_mode="average", num_recycles=0,
               use_templates=False, use_msa=False, best_metric="loss",
               crop_len=None, crop=False, prob_rest=0.5,
               subbatch_size=None, debug=False,
               use_alphafold=True, use_openfold=False, bfloat=True,
               loss_callback=None, data_dir="."):
    
    # fixbb: fix backbone design
    # hallucination: hallucinate protein structure based on nothing
    # binder: design binder based on receptor
    # partial: hallucinate protein based on partial structure
    # condition: hallucinate protein structure under provided msa, template, and restraints
    assert protocol in ["fixbb","hallucination","binder","partial","dock"]
    assert recycle_mode in ["average","add_prev","backprop","last","sample"]
    
    # decide if templates should be used
    if protocol == "binder": use_templates = True
    if protocol == "dock": use_templates = True

    self.protocol = protocol
    self._loss_callback = loss_callback
    self._num = num_seq
    self._copies = 1
    crop = False if crop_len is None else True
    self._args = {"use_templates":use_templates,
                  "use_msa": use_msa,
                  "recycle_mode":recycle_mode,
                  "debug":debug, "repeat":False,
                  "best_metric":best_metric,
                  'use_alphafold':use_alphafold,
                  'use_openfold':use_openfold,
                  "crop_len":crop_len,
                  "crop":crop,
                  "prob_rest":prob_rest,
                  "use_dgram":False}
    
    self.opt = {"dropout":True, "lr":1.0, "use_pssm":False,
                "recycles":num_recycles, "models":num_models, "sample_models":sample_models,
                "DM": {"thres": 10.0}, "masks": None, "num_msa": 64,
                "temp":1.0, "soft":0.0, "hard":0.0, "bias":0.0, "alpha":1.0,
                "con":      {"num":2, "cutoff":14.0, "binary":False, "seqsep":9},
                "i_con":    {"num":1, "cutoff":20.0, "binary":False},                 
                "template": {"aatype":21, "dropout":0.0},
                "weights":  {"helix":0.0, "plddt":0.0, "pae":0.0},
                "weights_bak": {"rest": 0.0, "rest_non": 0.0},
                "rest":{"cutoff":8.0, "binary":True, "entropy":True, "num":1},
                "rest_non":{"cutoff":8.0, "binary":True, "entropy":True, "num":1},
                "num_MvN": [1],}
    self.key = Key().get

    self.params = {}

    #############################
    # configure AlphaFold
    #############################
    # decide which config to use
    if use_templates:
      model_name = "model_1_ptm"
      self.opt["models"] = min(num_models, 2)
    else:
      model_name = "model_3_ptm"
    
    # generator config
    cfg = config.model_config(model_name)
    cfg.model.global_config.use_remat = True  
    cfg.model.global_config.subbatch_size = subbatch_size
    cfg.model.global_config.bfloat16 = bfloat
    cfg.model.global_config.use_dgram = self._args["use_dgram"]

    # number of recycles
    if recycle_mode == "average": num_recycles = 0
    cfg.model.num_recycle = num_recycles

    # initialize runner
    self._runner = model.RunModel(cfg, recycle_mode=recycle_mode)

    # load model_params
    model_names = []
    if use_templates:
      model_names += [f"model_{k}_ptm" for k in [1,2]]
      # model_names += [f"openfold_model_ptm_{k}" for k in [1,2]]    
    else:
      model_names += [f"model_{k}_ptm" for k in [1,2,3,4,5]]
      # model_names += [f"openfold_model_ptm_{k}" for k in [1,2]] + ["openfold_model_no_templ_ptm_1"]

    self._model_params, self._model_names = [],[]
    for model_name in model_names:
      params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir, fuse=True)
      if params is not None:
        if not use_templates:
          params = {k:v for k,v in params.items() if "template" not in k}
        self._model_params.append(params)
        self._model_names.append(model_name)

    # define gradient function
    self._grad_fn, self._fn = [jax.jit(x) for x in self._get_model()]
    # self._grad_fn, self._fn = self._get_model()

    #####################################
    # set protocol specific functions
    #####################################
    self.prep_inputs = self._prep_dock
    self._get_loss   = self._loss_dock
  
  def gen_infer(self, root_path):
    self.gen_pdb, self.gen_measures = [], []
    steps = len(self.gen_inputs)
    for i in tqdm(range(steps)):
      i_seqs = self.gen_inputs[i]['seq']
      i_seqs = jax.tree_util.tree_map(jnp.asarray, i_seqs)

      # update sequence features
      if self.protocol == "dock" and self._args["use_msa"]:
        update_seq(i_seqs["pseudo"], self._inputs, msa_input=self.feature_msa)
      else:
        pssm = jnp.where(self.opt["use_pssm"], i_seqs["pssm"], i_seqs["pseudo"])
        update_seq(i_seqs["pseudo"], self._inputs, seq_pssm=pssm)

      i_outputs = self._runner.apply(self._model_params[0], self.key(), self._inputs)
      i_aux = {"atom_positions":i_outputs["structure_module"]["final_atom_positions"],
               "atom_mask":i_outputs["structure_module"]["final_atom_mask"],                  
               "residue_index":self._inputs["residue_index"], "aatype":self._inputs["aatype"],
               "plddt":get_plddt(i_outputs), "pae":get_pae(i_outputs), "ptm":get_ptm(i_outputs)}
      i_aux = jax.tree_util.tree_map(np.asarray, i_aux)
      save_path = root_path + f'_{i+1}.pdb'
      save_pdb_from_aux(i_aux, save_path)
      self.gen_pdb.append({'atom_positions': i_aux['atom_positions'],
                           'atom_mask': i_aux['atom_mask']})
      # save iptm
      iptm = float(cal_iptm(i_aux['ptm'], self.asym_id))
      plddt = i_aux['plddt']
      self.gen_measures.append([iptm, plddt])

  def dis_infer(self, root_path):
    self.dis_measures = []
    steps = len(self.gen_outputs)
    aatype = jax.nn.one_hot(self._wt_aatype, 20)[None]

    for i in tqdm(range(steps)):
      # recover WT seq by 0.5 probability
      prob_seq = np.random.rand()
      aatype_ = aatype if prob_seq < 0.5 else jnp.asarray(self.gen_inputs[i]['seq']['pseudo'])
      if self.protocol == "dock" and self._args["use_msa"]:
        update_seq(aatype_, self._inputs, msa_input=self.feature_msa)
      else:
        update_seq(aatype_, self._inputs)
      igen_outputs = self.gen_outputs[i]
      # update template
      atom_position = igen_outputs["atom_positions"]
      atom_mask = igen_outputs["atom_mask"]
      self._update_template_complex(self._inputs, atom_position, atom_mask)
      idis_outputs = self._runner.apply(self._model_params[0], self.key(), self._inputs)
      i_aux = {"atom_positions":idis_outputs["structure_module"]["final_atom_positions"],
               "atom_mask":idis_outputs["structure_module"]["final_atom_mask"],
               "residue_index":self._inputs["residue_index"], "aatype":self._inputs["aatype"],
               "plddt":get_plddt(idis_outputs), "pae":get_pae(idis_outputs),"ptm": get_ptm(idis_outputs)}
      i_aux = jax.tree_util.tree_map(np.asarray, i_aux)
      save_path = root_path + f'_{i+1}.pdb'
      save_pdb_from_aux(i_aux, save_path)
      # save iptm
      iptm = float(cal_iptm(i_aux['ptm'], self.asym_id))
      plddt = i_aux['plddt']
      self.dis_measures.append([iptm, plddt])

  
  def _get_model(self, callback=None):

    # setup function to get gradients
    def _model(params, model_params, inputs, batch, key, opt):

      aux = {}
      key = Key(key=key).get

      #######################################################################
      # INPUTS
      #######################################################################
      # get sequence
      seq = self._get_seq(params, opt, aux, key())
            
      # update sequence features
      if self.protocol == "dock" and self._args["use_msa"]:
        update_seq(seq["pseudo"], inputs, msa_input=self.feature_msa)
      else:
        pssm = jnp.where(opt["use_pssm"], seq["pssm"], seq["pseudo"])
        update_seq(seq["pseudo"], inputs, seq_pssm=pssm)
      
      L = inputs["aatype"].shape[0]

      aux.update({"inputs": {"seq": {"pseudo": seq["pseudo"], "pssm": seq["pssm"]}}})
      # crop inputs
      if opt["crop_pos"].shape[0] < L:
        inputs = crop_feat(inputs, opt["crop_pos"], self._runner, add_batch=False)
        batch = crop_feat(batch, opt["crop_pos"], self._runner, add_batch=False)

      outputs = self._runner.apply(model_params, key(), inputs)
      aux["pdb"] = {"atom_positions":outputs["structure_module"]["final_atom_positions"],
                    "atom_mask":outputs["structure_module"]["final_atom_mask"],                  
                    "residue_index":inputs["residue_index"], "aatype":inputs["aatype"],
                    "plddt":get_plddt(outputs), "pae":get_pae(outputs),"ptm":get_ptm(outputs)}
      aux.update(aux["pdb"])

      # add aux outputs
      if self._args["recycle_mode"] == "average": aux["prev"] = outputs["prev"]
      if self._args["debug"]: aux["debug"] = {"inputs":inputs, "outputs":outputs, "opt":opt}

      #######################################################################
      # LOSS
      #######################################################################
      aux["losses"] = {}
      self._get_loss(inputs=inputs, outputs=outputs, opt=opt, aux=aux, batch=batch)

      if self._loss_callback is not None:
        aux["losses"].update(self._loss_callback(inputs, outputs, opt))

      # weighted loss
      w = opt["weights"]
      loss = sum([v * w[k] if k in w else v for k,v in aux["losses"].items()])
            
      return loss, aux
    
    return jax.value_and_grad(_model, has_aux=True, argnums=0), _model