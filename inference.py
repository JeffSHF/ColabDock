import jax
import numpy as np
import jax.numpy as jnp
import joblib

from colabdesign.af.prep import prep_pdb, prep_input_features
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.alphafold.model import config, model, data
from colabdesign.af.loss import get_plddt, get_pae, get_ptm
from colabdesign.af.utils import save_pdb_from_aux
from colabdesign.shared.utils import Key


class AF2_infer():
    def __init__(self,
                 num_recycles=3,
                 subbatch_size=None,
                 bfloat=True,
                 data_dir='./params'):
        self.data_dir = data_dir
        cfg = config.model_config('model_1_ptm')
        cfg.model.global_config.use_remat = True  
        cfg.model.global_config.subbatch_size = subbatch_size
        cfg.model.global_config.bfloat16 = bfloat
        cfg.model.global_config.use_dgram = False
        cfg.model.num_recycle = num_recycles
        self._runner = model.RunModel(cfg)
        self.key = Key().get

        self._model_params, self._model_names = [],[]
        model_names = [f"model_{k}_ptm" for k in [1,2]]
        for model_name in model_names:
            params = data.get_model_haiku_params(model_name=model_name,
                                                 data_dir=self.data_dir,
                                                 fuse=True)
            if params is not None:
                self._model_params.append(params)
                self._model_names.append(model_name)
    
    def setup(self,
              template_path,
              chains) -> None:
        self.template_path = template_path
        self.chains = chains
    
    def update_inputs(self,
                      target_seq='wt',
                      msa_seq='wt',
                      template_seq='wt',
                      rm_template_seq=True):
        self.rm_template_seq = rm_template_seq
        tmp_obj = prep_pdb(self.template_path,
                           chain=self.chains,
                           for_alphafold=False)
        self._wt_aatype = tmp_obj['batch']['aatype']
        seq_wt = ''.join([residue_constants.restypes[ind] for ind in self._wt_aatype])
        self.L = len(seq_wt)
        self.inputs = prep_input_features(self.L, N=1, T=1, eN=1)
        self.inputs = jax.tree_map(lambda x:jnp.array(x), self.inputs)

        chains = [c.strip() for c in self.chains.split(",")]
        self.lens = [(tmp_obj["idx"]["chain"] == c).sum() for c in chains]

        # update seq related
        # target seq
        seq_target = convert_seq(target_seq, seq_wt)
        seq_target = jax.nn.one_hot(seq_target, 22)[None]
        self._update_target_seq(seq_target)

        # msa seq
        seq_msa = convert_seq(msa_seq, seq_wt)
        seq_msa = jax.nn.one_hot(seq_msa, 22)[None]
        self._update_msa_seq(seq_msa)

        # update aatype
        self._update_aatype(self._wt_aatype)

        # update template
        pdb_pos = tmp_obj['batch']['all_atom_positions']
        pdb_pos = np.zeros([self.L, 37, 3])
        self.inputs["prev"] = {'prev_msa_first_row': np.zeros([self.L, 256]),
                                'prev_pair': np.zeros([self.L, self.L, 128]),
                                'prev_pos': pdb_pos}
        # template seq
        seq_tmp = convert_seq(template_seq, seq_wt)
        self._update_template(tmp_obj, seq_tmp)

        # update residue idx
        if len(self.lens) > 1:
            boundaries = [0] + list(np.cumsum(self.lens))
            residue_index = np.array(self.inputs['residue_index'])
            for ith in range(len(boundaries)-1):
                residue_index[boundaries[ith]:boundaries[ith+1]] += 50*ith
            self.inputs['residue_index'] = jnp.array(residue_index)
        else:
            pass
            # raise Exception('current is only suitable for complex!')

        # update others
        self.inputs["mask_template_interchain"] = False
        self.inputs["use_dropout"] = False
    
    def _update_template(self, tmp_obj, seq_tmp):
        ''''dynamically update template features'''
        # aatype = is used to define template's CB coordinates (CA in case of glycine)
        # template_aatype = is used as template's sequence
        batch = tmp_obj['batch']
        aatype = self._wt_aatype
        template_aatype = seq_tmp
        
        # get pseudo-carbon-beta coordinates (carbon-alpha for glycine)
        pb, pb_mask = model.modules.pseudo_beta_fn(aatype,
                                                   batch["all_atom_positions"],
                                                   batch["all_atom_mask"])
        template_dgram = model.modules.dgram_from_positions(pb, **self._runner.config.model.embeddings_and_evoformer.template.dgram_features)
        self.inputs['template_dgram'] = template_dgram[None]
        self.inputs['template_mask_2d'] = jnp.ones([self.L, self.L])[None]
        self.inputs['template_aatype'] = template_aatype[None]
        self.inputs['template_all_atom_positions'] = self.inputs['template_all_atom_positions'].at[0].set(batch["all_atom_positions"])
        self.inputs['template_all_atom_mask'] = self.inputs['template_all_atom_mask'].at[0].set(batch["all_atom_mask"])
    
        if self.rm_template_seq:
            self.inputs['template_all_atom_mask'] = self.inputs['template_all_atom_mask'].at[:,:,5:].set(0)
    
    def _update_target_seq(self, seq):
        '''update the sequence features'''
        if seq.ndim == 3:
            target_feat = jnp.zeros_like(self.inputs["target_feat"]).at[...,:20].set(seq[0,...,:20])
        else:
            target_feat = jnp.zeros_like(self.inputs["target_feat"]).at[...,:20].set(seq[...,:20])
        self.inputs.update({"target_feat":target_feat})

    def _update_msa_seq(self, seq, seq_1hot=None, seq_pssm=None, msa_input=None):
        '''update the sequence features'''
        if seq_1hot is None: seq_1hot = seq 
        if seq_pssm is None: seq_pssm = seq
        
        seq_1hot = jnp.pad(seq_1hot,[[0,0],[0,0],[0,22-seq_1hot.shape[-1]]])
        seq_pssm = jnp.pad(seq_pssm,[[0,0],[0,0],[0,22-seq_pssm.shape[-1]]])
        if msa_input is None:
            msa_feat = jnp.zeros_like(self.inputs["msa_feat"]).at[0, :, :22].set(seq_1hot[0]).at[0, :, 25:47].set(seq_pssm[0])
        else:
            msa_feat = jnp.array(msa_input['msa_feat'][0]).at[0, :, :22].set(seq_1hot[0])
        self.inputs.update({"msa_feat":msa_feat})
    
    def _update_aatype(self, aatype):
        r = residue_constants
        a = {"atom14_atom_exists":r.restype_atom14_mask,
             "atom37_atom_exists":r.restype_atom37_mask,
             "residx_atom14_to_atom37":r.restype_atom14_to_atom37,
             "residx_atom37_to_atom14":r.restype_atom37_to_atom14}
        mask = self.inputs["seq_mask"][:,None]
        self.inputs.update(jax.tree_map(lambda x:jnp.where(mask,jnp.asarray(x)[aatype],0), a))
        self.inputs["aatype"] = aatype

    def run_infer(self, save_path):
        outputs = self._runner.apply(self._model_params[0], self.key(), self.inputs)
        aux = {"atom_positions":outputs["structure_module"]["final_atom_positions"],
                 "atom_mask":outputs["structure_module"]["final_atom_mask"],                  
                 "residue_index":self.inputs["residue_index"], "aatype":self.inputs["aatype"],
                 "plddt":get_plddt(outputs), "pae":get_pae(outputs), "ptm":get_ptm(outputs)}
        aux = jax.tree_util.tree_map(np.asarray, aux)
        save_pdb_from_aux(aux, save_path)

def convert_seq(seq, wt,
                mapping=residue_constants.restypes_with_x + ['-']):
    # wt(str)
    # map
    if seq == 'wt':
        seq = wt
    elif type(seq) == str and len(seq) == 1:
        seq = seq * len(wt)
    elif type(seq) == str and len(seq) == len(wt):
        pass
    else:
        raise Exception('error')
    return np.array([mapping.index(iaa) for iaa in seq])


if __name__ == '__main__':
    AF_model = AF2_infer(bfloat=False)
    AF_model.setup(template_path='./protein/4HFF/PDB/4HFF.pdb',
                   chains='A,B')
    AF_model.update_inputs(target_seq='X',
                           msa_seq='X',
                           template_seq='X',
                           rm_template_seq=True)
    AF_model.run_infer('./results/pred.pdb')

