from pathlib import Path

import torch
from allatom_design.checkpoint_utils import repair_state_dict
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from fampnn.model.sd_model import SeqDenoiser

in_ckpt_path = "/media/scratch/huang_lab/allatom_design/allatom_design/c0_aasd4/post_hoc_ema_ckpts/ema-step300000-std0.250.ckpt"
out_ckpt_path = "weights/fampnn_0_0.pt"
# in_ckpt_path = "/media/scratch/huang_lab/allatom_design/allatom_design/c0_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.010.ckpt"
# out_ckpt_path = "weights/fampnn_0_3.pt"
# in_ckpt_path = "/media/scratch/huang_lab/allatom_design/allatom_design/b13_aasd6/checkpoints/sd-step100000-epoch300.ckpt"
# out_ckpt_path = "weights/fampnn_0_3_cath.pt"

lit_sd_model = LitSeqDenoiser.load_from_checkpoint(in_ckpt_path).eval()

# Model config
model_cfg = lit_sd_model.cfg.model

# State dict
# get rid of torch.compile
state_dict = repair_state_dict(lit_sd_model.model.state_dict())

# get rid of bb std and mean
state_dict.pop("bb_mean")
state_dict.pop("bb_std")

# get rid of certain unused confidence module params
unused_keys = ["denoiser.scn_diffusion_module.confidence_module.structure_encoder.sidechain_features.embeddings.linear.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.sidechain_features.embeddings.linear.bias", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.sidechain_features.edge_embedding.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.sidechain_features.norm_edges.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.sidechain_features.norm_edges.bias", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.features.embeddings.linear.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.features.embeddings.linear.bias", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.features.edge_embedding.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.features.norm_edges.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.features.norm_edges.bias", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.W_e.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.W_e.bias", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.W_s.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.W_e2.weight", "denoiser.scn_diffusion_module.confidence_module.structure_encoder.W_e2.bias"]
for key in unused_keys:
    state_dict.pop(key)

# Save
Path(out_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": state_dict, "model_cfg": model_cfg}, out_ckpt_path)

# Test loading
ckpt = torch.load(out_ckpt_path)
model = SeqDenoiser(ckpt["model_cfg"]).eval()
model.load_state_dict(ckpt["state_dict"])

print("DONE")
