from typing import Optional, Union
import torch
from safetensors.torch import load_file, save_file

def load_safetensors(path: str, device: Union[str, torch.device]):
    try:
        return load_file(path, device=device)
    except:
        return load_file(path)  # prevent device invalid Error
    
model_path = "C:\ComfyUIModel\models\checkpoints\copaxTimelessxl_xplusPoses.safetensors"
save_path = "C:\ComfyUIModel\models\checkpoints\copaxTimelessxl_xplusPoses_onlySD.safetensors"
device = "cuda:0"
sd = load_safetensors(model_path, device=device)

# cover  copaxTimelessxl
sd = {k: v for k, v in sd.items() if "text_encoders" not in k}
sd = {k: v for k, v in sd.items() if "vae" not in k}
sd = {k.replace("diffusion_", ""): v for k, v in sd.items()}
sd = {k.replace("model.", ""): v for k, v in sd.items()}


sd = {k: v.to(device, dtype=torch.bfloat16) for k, v in sd.items()}


#save the model with safetensors
save_file(sd, save_path)
