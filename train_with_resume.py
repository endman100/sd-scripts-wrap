import sys
# sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import os
import subprocess
import toml

venv_activate_path = os.path.abspath('./venv/Scripts/activate')
if not os.path.exists(venv_activate_path):
    venv_activate_path = ""
else:
    venv_activate_path += " && "

def get_command(initial_epoch, resume, **kwargs):
    pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", r"C:\ComfyUIModel\models\checkpoints\flux1-dev.safetensors")
    clip_l = kwargs.get("clip_l", r"C:\ComfyUIModel\models\clip\clip_l.safetensors")
    t5xxl = kwargs.get("t5xxl", r"C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors")
    ae = kwargs.get("ae", r"C:\ComfyUIModel\models\vae\ae.safetensors")
    wandb_dir = kwargs.get("wandb_dir", r"./wandb")
    max_train_epochs = kwargs.get("max_train_epochs", 20)
    learning_rate = kwargs.get("learning_rate", 1e-4)
    network_dim = kwargs.get("network_dim", 16)
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]
    accumulation_steps = kwargs["accumulation_steps"]

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path,"sd-scripts", "flux_train_network.py")


    if resume != "":
        keep_cmd = f'\
        {venv_activate_path}accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --clip_l="{clip_l}" \
            --t5xxl="{t5xxl}" \
            --ae="{ae}" \
            --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module networks.lora_flux --network_dim={network_dim} --optimizer_type adamw8bit --learning_rate={learning_rate} \
            --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs=1 --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --timestep_sampling="faster" --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
            --initial_epoch={initial_epoch + 1} --skip_until_initial_step \
            --resume="{resume}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="fun" --log_tracker_name="fun lora resume train" \
            --lowram --save_state --accumulation_steps="{accumulation_steps}"'
    else:
        keep_cmd = f'\
        {venv_activate_path}accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
            --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
            --clip_l="{clip_l}" \
            --t5xxl="{t5xxl}" \
            --ae="{ae}" \
            --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module networks.lora_flux --network_dim={network_dim} --optimizer_type adamw8bit --learning_rate={learning_rate} \
            --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs=1 --dataset_config="{dataset_config}" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --log_with wandb --logging_dir="{wandb_dir}" --wandb_run_name="fun" --log_tracker_name="fun lora resume train" \
            --timestep_sampling="faster" --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
            --lowram --save_state --accumulation_steps="{accumulation_steps}"'
    return keep_cmd

def create_toml_file(**kwargs):
    resolution = kwargs.get("resolution", 1024)
    batch_size = kwargs.get("batch_size", 2)
    train_dir = kwargs.get("train_dir")
    num_repeats = kwargs.get("num_repeats", 1)
    caption_extension = "txt"
    toml_path = kwargs.get("toml_path")
    class_tokens = kwargs.get("class_tokens")
    
    with open(toml_path, "w") as f:
        data = {
            "datasets": [
                {
                    "resolution": resolution,
                    "batch_size": 2, #4090
                    "subsets": [
                        {
                            "image_dir": train_dir,  # dataset_images
                            "class_tokens": class_tokens,
                            "num_repeats": num_repeats,
                            "caption_extension": caption_extension
                        }
                    ]
                }
            ]
        }
        toml.dump(data, f)

def train_with_resume(output_name, output_dir, wandb_dir, **kwargs):
    kwargs["class_tokens"] = output_name
    toml_path = kwargs.get("toml_path")
    batch_size = kwargs.get("batch_size", 2) #4090
    kwargs["accumulation_steps"] = batch_size // 2

    create_toml_file(**kwargs)

    log_path = os.path.join(output_dir, "train.log")
    while(True):
        print("run_command")
        dir_path = os.path.dirname(log_path)
        if not os.path.exists(dir_path) and dir_path != "":
            os.makedirs(dir_path)
        
        #檢查是否有儲存點
        max_epoch = -1
        max_resume = ""
        has_end = False
        if(os.path.exists(output_dir)):
            for output_file in os.listdir(output_dir):
                output_filepath = os.path.join(output_dir, output_file)
                print(output_filepath)
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
                new_end_path = os.path.join(output_dir, f"{output_name}-{str(max_epoch + 1).zfill(6)}.safetensors")
                os.rename(end_path, new_end_path)
                
                end_dir_path = os.path.join(output_dir, f"{name}-state")
                new_end_dir_path = os.path.join(output_dir, f"{output_name}-{str(max_epoch + 1).zfill(6)}-state")
                os.rename(end_dir_path, new_end_dir_path)
                max_epoch += 1

        print(f"max_epoch: {max_epoch}, max_resume: {max_resume}", flush=True)
        kwargs["output_name"] = output_name
        kwargs["dataset_config"] = toml_path
        kwargs["output_dir"] = output_dir
        kwargs["wandb_dir"] = wandb_dir
        cmd = get_command(max_epoch, max_resume, **kwargs)
        with open(log_path, "a", encoding="utf-8") as f:
            print(f"run_command: {cmd}")
            # process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f, text=True)
            process = subprocess.Popen(cmd, shell=True, text=True)
            process.wait()
        
        if process.returncode != 0:
            print(f"run_command error: {cmd}")
            print(f"run again")
            workspace = os.path.dirname(output_dir)
            interrpt_path = os.path.join(workspace, "interrupt.json")
            if os.path.exists(interrpt_path):
                with open(interrpt_path, "r") as f:
                    interrupt_json = json.load(f)
                if(interrupt_json["interrupt"] == True):
                    print("run_command interrupted")
            time.sleep(60)
        else:
            break

if __name__ == "__main__":
    output_name = "flux1dev"
    toml_path = r"D:\symbolCopyResult\result\_lora_train\LilyLinglanv2Test\config.toml"
    output_dir = r"D:\symbolCopyResult\result\_lora_train\LilyLinglanv2Test\models"
    wandb_dir = r"D:\symbolCopyResult\result\_lora_train\LilyLinglanv2Test\wandb"
    train_dir = r"D:/symbolCopyResult/result\\_lora_train\\LilyLinglanv2Test\\train"
    kwargs = {
        "resolution": 1024,
        "batch_size": 8,
        "train_dir": train_dir,
        "num_repeats": 1,
        "toml_path": toml_path,
    }
    train_with_resume(output_name, output_dir, wandb_dir, **kwargs)