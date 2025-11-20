import sys
# sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import os
import subprocess
import toml
py_dir = os.path.dirname(os.path.abspath(__file__))
venv_activate_path = os.path.join(py_dir, "..", "..", "..", "./.venv/Scripts/activate")
if not os.path.exists(venv_activate_path):
    venv_activate_path = ""
    print("No venv found, use system python")
    exit(1)
else:
    venv_activate_path += " && "

def get_command(initial_epoch, resume, **kwargs):
    pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", r"C:\ComfyUIModel\models\checkpoints\bluePencilFlux1_v021.safetensors")
    clip_l = kwargs.get("clip_l", r"C:\ComfyUIModel\models\clip\clip_l.safetensors")
    t5xxl = kwargs.get("t5xxl", r"C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors")
    ae = kwargs.get("ae", r"C:\ComfyUIModel\models\vae\ae.safetensors")
    wandb_dir = kwargs.get("wandb_dir", r"./wandb")
    max_train_epochs = kwargs.get("max_train_epochs", 6)
    learning_rate = kwargs.get("learning_rate", 1e-4)
    network_dim = kwargs.get("network_dim", 16)
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]
    save_every_n_epochs = kwargs.get("save_every_n_epochs", 1)
    gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path,"sd-scripts", "flux_train_network.py")
    
    if(initial_epoch == max_train_epochs):
        return ""

    if resume != "":
        keep_cmd = f'\
        {venv_activate_path}accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --clip_l="{clip_l}" \
            --t5xxl="{t5xxl}" \
            --ae="{ae}" \
            --enable_bucket --max_bucket_reso=1280 \
            --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module networks.lora_flux --network_dim={network_dim} --optimizer_type adamw8bit --learning_rate={learning_rate} \
            --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --timestep_sampling="faster" --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
            --initial_epoch={initial_epoch + 1} --skip_until_initial_step \
            --resume="{resume}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="fun" --log_tracker_name="fun lora resume train" \
            --lowram --save_state --gradient_accumulation_steps="{gradient_accumulation_steps}"'
    else:
        keep_cmd = f'\
        {venv_activate_path}accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --clip_l="{clip_l}" \
            --t5xxl="{t5xxl}" \
            --ae="{ae}" \
            --enable_bucket --max_bucket_reso=1280 \
            --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module networks.lora_flux --network_dim={network_dim} --optimizer_type adamw8bit --learning_rate={learning_rate} \
            --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="fun" --log_tracker_name="fun lora resume train" \
            --timestep_sampling="faster" --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
            --lowram --save_state --gradient_accumulation_steps="{gradient_accumulation_steps}"'
    return keep_cmd

def get_command_sdxl(initial_epoch, resume, **kwargs):
    pretrained_model_name_or_path =  r"C:\ComfyUIModel\models\checkpoints\waiNSFWIllustrious_v140.safetensors"
    wandb_dir = kwargs.get("wandb_dir", r"./wandb")
    max_train_epochs = kwargs.get("max_train_epochs", 6)
    learning_rate = kwargs.get("learning_rate", 1e-4)
    network_dim = kwargs.get("network_dim", 16)
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]
    network_module = kwargs.get("network_module", "networks.lora")
    save_every_n_epochs = kwargs.get("save_every_n_epochs", 5)
    gradient_accumulation_steps = 1

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path,"sd-scripts", "sdxl_train_network.py")

    if resume != "":
        keep_cmd = f'\
        {venv_activate_path}accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} \
            --cache_latents_to_disk --cache_text_encoder_outputs_to_disk \
            --network_train_unet_only \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --initial_epoch={initial_epoch + 1} --skip_until_initial_step \
            --resume="{resume}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="MeifeiTest" --log_tracker_name="fun lora_fa resume train" \
            --save_state '
    else:
        keep_cmd = f'\
        {venv_activate_path}accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} \
            --cache_latents_to_disk --cache_text_encoder_outputs_to_disk \
            --network_train_unet_only  \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="MeifeiTest" --log_tracker_name="fun lora_fa resume train" \
            --save_state '
    return keep_cmd

def get_command_sdxl_clip(initial_epoch, resume, **kwargs):
    pretrained_model_name_or_path =  r"C:\ComfyUIModel\models\checkpoints\waiNSFWIllustrious_v140.safetensors"
    wandb_dir = kwargs.get("wandb_dir", r"./wandb")
    max_train_epochs = kwargs.get("max_train_epochs", 6)
    learning_rate = kwargs.get("learning_rate", 1e-4)
    text_encoder_lr = kwargs.get("text_encoder_lr", 1e-5)
    network_dim = 16
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]
    network_module = kwargs.get("network_module", "networks.lora")
    save_every_n_epochs = kwargs.get("save_every_n_epochs", 5)
    gradient_accumulation_steps = 1

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path,"sd-scripts", "sdxl_train_network.py")

    if resume != "":
        keep_cmd = f'\
        {venv_activate_path}accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} --text_encoder_lr={text_encoder_lr} \
            --cache_latents_to_disk \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --initial_epoch={initial_epoch + 1} --skip_until_initial_step \
            --resume="{resume}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="MeifeiTest" --log_tracker_name="fun lora_fa resume train" \
            --save_state '
    else:
        keep_cmd = f'\
        {venv_activate_path}accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} --text_encoder_lr={text_encoder_lr} \
            --cache_latents_to_disk \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="MeifeiTest" --log_tracker_name="fun lora_fa resume train" \
            --save_state '
    return keep_cmd

def get_command_sdxl_suffle(initial_epoch, resume, **kwargs):
    if os.name == 'nt':
        pretrained_model_name_or_path =  r"C:\ComfyUIModel\models\checkpoints\waiNSFWIllustrious_v140.safetensors"
    else:
        pretrained_model_name_or_path =  "/workspace/storage/stable_diffusion/models/ckpt/waiNSFWIllustrious_v140.safetensors"
    wandb_dir = kwargs.get("wandb_dir", r"./wandb")
    max_train_epochs = kwargs.get("max_train_epochs", 6)
    learning_rate = kwargs.get("learning_rate", 1e-4)
    network_dim = kwargs.get("network_dim", 16)
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]
    network_module = kwargs.get("network_module", "networks.lora")
    save_every_n_epochs = kwargs.get("save_every_n_epochs", 5)
    caption_tag_dropout_rate = kwargs.get("caption_tag_dropout_rate", 0.2)
    gradient_accumulation_steps = 1

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path,"sd-scripts", "sdxl_train_network.py")

    if resume != "":
        keep_cmd = f'\
        {venv_activate_path}accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} \
            --cache_latents_to_disk \
            --network_train_unet_only \
            --shuffle_caption --keep_tokens=1 --caption_tag_dropout_rate={caption_tag_dropout_rate} \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --initial_epoch={initial_epoch + 1} --skip_until_initial_step \
            --resume="{resume}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="MeifeiTest" --log_tracker_name="fun lora_fa resume train" \
            --save_state '
    else:
        keep_cmd = f'\
        {venv_activate_path}accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} \
            --cache_latents_to_disk \
            --shuffle_caption --keep_tokens=1 --caption_tag_dropout_rate={caption_tag_dropout_rate} \
            --network_train_unet_only  \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="MeifeiTest" --log_tracker_name="fun lora_fa resume train" \
            --save_state '
    return keep_cmd

def get_command_sdxl_all(initial_epoch, resume, **kwargs):
    if os.name == 'nt':
        pretrained_model_name_or_path =  r"C:\ComfyUIModel\models\checkpoints\waiNSFWIllustrious_v140.safetensors"
    else:
        pretrained_model_name_or_path =  "/workspace/storage/stable_diffusion/models/ckpt/waiNSFWIllustrious_v140.safetensors"
    wandb_dir = kwargs.get("wandb_dir", r"./wandb")
    max_train_epochs = kwargs.get("max_train_epochs", 6)
    learning_rate = kwargs.get("learning_rate", 1e-4)
    text_encoder_lr = kwargs.get("text_encoder_lr", 1e-5)
    network_dim = kwargs.get("network_dim", 16)
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]
    network_module = kwargs.get("network_module", "networks.lora")
    save_every_n_epochs = kwargs.get("save_every_n_epochs", 5)
    caption_tag_dropout_rate = kwargs.get("caption_tag_dropout_rate", 0.2)
    gradient_accumulation_steps = 1

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path,"sd-scripts", "sdxl_train_network.py")

    if resume != "":
        keep_cmd = f'\
        {venv_activate_path}accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} --text_encoder_lr={text_encoder_lr} \
            --cache_latents_to_disk \
            --enable_bucket \
            --shuffle_caption --keep_tokens=1 --caption_tag_dropout_rate={caption_tag_dropout_rate} \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --initial_epoch={initial_epoch + 1} --skip_until_initial_step \
            --resume="{resume}" \
            --save_state '
    else:
        keep_cmd = f'\
        {venv_activate_path}accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module={network_module} --network_dim={network_dim} --optimizer_type AdamW --learning_rate={learning_rate} --text_encoder_lr={text_encoder_lr} \
            --cache_latents_to_disk \
            --enable_bucket \
            --shuffle_caption --keep_tokens=1 --caption_tag_dropout_rate={caption_tag_dropout_rate} \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs={save_every_n_epochs} --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --save_state '
    return keep_cmd

def get_command_qwen(initial_epoch, resume, **kwargs):
    if os.name == 'nt':
        pretrained_model_name_or_path =  r"C:\ComfyUIModel\models\checkpoints\Gazai-Style-000016.safetensors"
        vae_path = r"C:\ComfyUIModel\models\vae\qwen_image_vae_train.safetensors"
        clip_path = r"C:\ComfyUIModel\models\clip\qwen_2.5_vl_7b.safetensors"
    else:
        pretrained_model_name_or_path =  "/home/gazai/models/checkpoints/Gazai-Style-000016.safetensors"
        vae_path = "/home/gazai/models/vae/qwen_image_vae_train.safetensors"
        clip_path = "/home/gazai/models/text_encoders/qwen_2.5_vl_7b.safetensors"
    wandb_dir = kwargs.get("wandb_dir", r"./wandb")
    max_train_epochs = kwargs.get("max_train_epochs", 6) - initial_epoch
    # max_train_epochs = 6
    learning_rate = kwargs.get("learning_rate", 1e-4)
    network_dim = kwargs.get("network_dim", 16)
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]
    network_module = kwargs.get("network_module", "networks.lora")
    # save_every_n_epochs = kwargs.get("save_every_n_epochs", 5)
    save_every_n_epochs = 1
    gradient_accumulation_steps = 2
    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path, "musubi-tuner", "qwen_image_train_network.py")
    cache_latents_py_path = os.path.join(py_dir_path, "musubi-tuner", "qwen_image_cache_latents.py")
    text_encoder_path = os.path.join(py_dir_path, "musubi-tuner", "qwen_image_cache_text_encoder_outputs.py")

    keep_cmds = []
    if resume != "":
        keep_cmd = f'\
            {venv_activate_path}accelerate launch --gpu_ids 0 --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 {train_py_path} \
                --dit "{pretrained_model_name_or_path}" \
                --vae "{vae_path}" \
                --text_encoder "{clip_path}" \
                --dataset_config "{dataset_config}" \
                --sdpa --mixed_precision bf16 \
                --timestep_sampling shift \
                --weighting_scheme none --discrete_flow_shift 2.2 \
                --optimizer_type adamw8bit --learning_rate {learning_rate} --gradient_checkpointing \
                --max_data_loader_n_workers 2 --persistent_data_loader_workers \
                --network_module networks.lora_qwen_image \
                --gradient_accumulation_steps {gradient_accumulation_steps} \
                --network_dim 16 \
                --fp8_base --fp8_vl --xformers \
                --resume="{resume}" \
                --log_with wandb --logging_dir="{wandb_dir}" --log_tracker_name="First train lora" --wandb_run_name="Qwen Image Lora" \
                --max_train_epochs {max_train_epochs} --save_every_n_epochs {save_every_n_epochs} --seed 42 \
                --output_dir "{output_dir}" --output_name "{output_name}" \
                --save_state '
        keep_cmds.append(keep_cmd)
    else:
        keep_cmd = f'\
            python {cache_latents_py_path} \
                --dataset_config="{dataset_config}" \
                --vae="{vae_path}" --skip_existing '
        keep_cmds.append(keep_cmd)
        
        keep_cmd = f'\
            python {text_encoder_path} \
                --dataset_config="{dataset_config}" \
                --text_encoder="{clip_path}" \
                --batch_size 1 --skip_existing '
        keep_cmds.append(keep_cmd)

        keep_cmd = f'\
            {venv_activate_path}accelerate launch --gpu_ids 0 --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 {train_py_path} \
                --dit "{pretrained_model_name_or_path}" \
                --vae "{vae_path}" \
                --text_encoder "{clip_path}" \
                --dataset_config "{dataset_config}" \
                --sdpa --mixed_precision bf16 \
                --timestep_sampling shift \
                --weighting_scheme none --discrete_flow_shift 2.2 \
                --optimizer_type adamw8bit --learning_rate {learning_rate} --gradient_checkpointing \
                --max_data_loader_n_workers 2 --persistent_data_loader_workers \
                --gradient_accumulation_steps {gradient_accumulation_steps} \
                --network_module networks.lora_qwen_image \
                --network_dim 16 \
                --fp8_base --fp8_vl --xformers \
                --log_with wandb --logging_dir="{wandb_dir}" --log_tracker_name="First train lora" --wandb_run_name="Qwen Image Lora" \
                --max_train_epochs {max_train_epochs} --save_every_n_epochs {save_every_n_epochs} --seed 42 \
                --output_dir "{output_dir}" --output_name "{output_name}" \
                --save_state '
        keep_cmds.append(keep_cmd)
    
    return keep_cmds

def create_toml_file(**kwargs):
    resolution = kwargs.get("resolution", 1024)
    batch_size = kwargs.get("batch_size", 2)
    train_dir = kwargs.get("train_dir")
    num_repeats = kwargs.get("num_repeats", 10)
    num_repeats_regularization = kwargs.get("num_repeats_regularization", 1)
    caption_extension = "txt"
    toml_path = kwargs.get("toml_path")
    class_tokens = kwargs.get("class_tokens")
    regularization_dir = kwargs.get("regularization_dir", None)
    cache_dir = kwargs.get("cache_dir", train_dir)
    cache_dir_regularization = kwargs.get("cache_dir_regularization", regularization_dir)
    is_qwen_format = kwargs.get("train_method") == "get_command_qwen"
    if is_qwen_format:
        data = {
            "general": {
                "resolution": resolution,
                "batch_size": 2,
                "enable_bucket": True,
                "bucket_no_upscale": False,
                "caption_extension": ".txt"
            },
            "datasets": [
                {
                    "image_directory": train_dir,
                    "cache_directory": cache_dir,
                    "num_repeats": num_repeats
                },
                {
                    "image_directory": regularization_dir,
                    "cache_directory": cache_dir_regularization,
                    "num_repeats": num_repeats_regularization
                }
            ]
        }
    else:
        data = {
            "datasets": [
                {
                    "resolution": resolution,
                    "batch_size": batch_size, #4090
                    "subsets": [
                        {
                            "image_dir": train_dir,  # dataset_images
                            "class_tokens": class_tokens,
                            "num_repeats": num_repeats,
                            "caption_extension": caption_extension
                        },
                    ]
                }
            ]
        }

        if regularization_dir is not None:
            data["datasets"][0]["subsets"].append(
                {
                    "image_dir": regularization_dir,
                    "class_tokens": "regularization",
                    "num_repeats": num_repeats_regularization,
                    "caption_extension": caption_extension,
                    "is_reg": True
                }
            )
    with open(toml_path, "w") as f:
        toml.dump(data, f)

def find_last_checkpoint(output_dir, output_name, save_every_n_epochs):
    #檢查是否有儲存點
    max_epoch = -1
    max_resume = ""
    has_end = False
    if(os.path.exists(output_dir)):
        print(f"output_dir: {output_dir}")
        print(f"output_dir: {os.listdir(output_dir)}")
        
        for output_file in os.listdir(output_dir):
            output_filepath = os.path.join(output_dir, output_file)
            print(f"check old checkpoint:{output_filepath}", flush=True)
            if(os.path.isdir(output_filepath)):
                if(len(output_file.split("-")) == 3):
                    name, epoch, state = output_file.split("-")
                    epoch = int(epoch)
                    if (max_epoch < epoch):
                        max_epoch = epoch
                        max_resume = output_filepath
                elif(len(output_file.split("-")) == 2):
                    has_end = True
        if has_end:
            end_path = os.path.join(output_dir, f"{name}.safetensors")
            max_epoch += save_every_n_epochs
            new_end_path = os.path.join(output_dir, f"{output_name}-{str(max_epoch).zfill(6)}.safetensors")
            os.rename(end_path, new_end_path)
            
            end_dir_path = os.path.join(output_dir, f"{name}-state")
            new_end_dir_path = os.path.join(output_dir, f"{output_name}-{str(max_epoch).zfill(6)}-state")
            os.rename(end_dir_path, new_end_dir_path)
    return max_epoch, max_resume

def train_with_resume(output_name, output_dir, wandb_dir, **kwargs):
    kwargs["class_tokens"] = kwargs.get("class_tokens", output_name)
    toml_path = kwargs.get("toml_path")
    batch_size = kwargs.get("batch_size", 2) #4090
    train_method = kwargs.get("train_method", "get_command_sdxl_suffle")
    cache_dir = kwargs.get("cache_dir", None)
    kwargs["gradient_accumulation_steps"] = batch_size // 2
    print(f"batch_size: {batch_size}, gradient_accumulation_steps: {kwargs['gradient_accumulation_steps']}")

    create_toml_file(**kwargs)

    log_path = os.path.join(output_dir, "train.log")
    while(True):
        print("run_command")
        dir_path = os.path.dirname(log_path)
        if not os.path.exists(dir_path) and dir_path != "":
            os.makedirs(dir_path)
        
        max_epoch, max_resume = find_last_checkpoint(output_dir, output_name, kwargs.get("save_every_n_epochs", 1))    
        print(f"max_epoch: {max_epoch}, max_resume: {max_resume}", flush=True)
        if(max_epoch >= kwargs.get("max_train_epochs", 10)):
            print(f"max_epoch:{max_epoch} >= max_train_epochs:{kwargs.get("max_train_epochs", 10)}, no need to train again")
            break
        
        kwargs["output_name"] = output_name
        kwargs["dataset_config"] = toml_path
        kwargs["output_dir"] = output_dir
        kwargs["wandb_dir"] = wandb_dir

        if train_method == "train_with_clip":
            cmd = get_command_sdxl_clip(max_epoch, max_resume, **kwargs)
        elif train_method == "get_command_sdxl":
            cmd = get_command_sdxl(max_epoch, max_resume, **kwargs)
        elif train_method == "get_command_sdxl_all":
            cmd = get_command_sdxl_all(max_epoch, max_resume, **kwargs)
        elif train_method == "get_command_sdxl_suffle":
            cmd = get_command_sdxl_suffle(max_epoch, max_resume, **kwargs)
        elif train_method == "get_command_qwen":
            if(max_epoch == -1):
                max_epoch = 0
            cmd = get_command_qwen(max_epoch, max_resume, **kwargs)
            
        with open(log_path, "a", encoding="utf-8") as f:
            if type(cmd) == str:
                cmd = [cmd]

            is_error = False
            for c in cmd:
                print(f"run_command: {c}", flush=True)
            
                # process = subprocess.Popen(c, shell=True, stdout=f, stderr=f, text=True)
                process = subprocess.Popen(c, shell=True, text=True)
                process.wait()
                if process.returncode != 0:
                    print(f"run_command error: {c}")
                    print(f"run again")
                    workspace = os.path.dirname(output_dir)
                    interrpt_path = os.path.join(workspace, "interrupt.json")
                    if os.path.exists(interrpt_path):
                        with open(interrpt_path, "r") as f:
                            interrupt_json = json.load(f)
                        if(interrupt_json["interrupt"] == True):
                            print("run_command interrupted")
                    time.sleep(60)
                    is_error = True
                    break
            if is_error:
                continue
            else:
                break
    print(f"train_with_resume end")

if __name__ == "__main__":
    output_name = "flux1dev"
    toml_path = r"D:\AICGCode\SymbolCopy\result\lora_train\Meifei\config.toml"
    output_dir = r"D:\AICGCode\SymbolCopy\result\lora_train\Meifei\models"
    wandb_dir = r"D:\AICGCode\SymbolCopy\result\lora_train\Meifei\wandb"
    train_dir = r"D:\AICGCode\SymbolCopy\result\lora_train\Meifei\\train"
    kwargs = {
        "resolution": 1024,
        "batch_size": 2,
        "train_dir": train_dir,
        "num_repeats": 100,
        "toml_path": toml_path,
    }
    train_with_resume(output_name, output_dir, wandb_dir, **kwargs)