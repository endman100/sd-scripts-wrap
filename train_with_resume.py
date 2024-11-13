import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import os
import subprocess


venv_activate_path = os.path.abspath('./venv/Scripts/activate')
def get_command(initial_epoch, resume, **kwargs):
    pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", r"C:\ComfyUIModel\models\checkpoints\flux1-dev.safetensors")
    clip_l = kwargs.get("clip_l", r"C:\ComfyUIModel\models\clip\clip_l.safetensors")
    t5xxl = kwargs.get("t5xxl", r"C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors")
    ae = kwargs.get("ae", r"C:\ComfyUIModel\models\vae\ae.safetensors")
    max_train_epochs = kwargs.get("max_train_epochs", 10)
    learning_rate = kwargs.get("learning_rate", 5e-5)
    network_dim = kwargs.get("network_dim", 16)
    dataset_config = kwargs["dataset_config"]
    output_dir = kwargs["output_dir"]
    output_name = kwargs["output_name"]

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    train_py_path = os.path.join(py_dir_path,"sd-scripts", "flux_train_network.py")


    if resume != "":
        keep_cmd = f'\
        {venv_activate_path} && accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
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
            --log_with wandb --wandb_run_name="fun" --log_tracker_name="fun lora resume train" \
            --lowram --save_state '
    else:
        keep_cmd = f'\
        {venv_activate_path} && accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 {train_py_path} \
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
            --log_with wandb --wandb_run_name="fun" --log_tracker_name="fun lora resume train" \
            --timestep_sampling="faster" --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
            --lowram --save_state '
    return keep_cmd

def train_with_resume(output_name, dataset_config, output_dir, **kwargs):
    log_path = os.path.join(output_dir, "train.log")
    while(True):
        print("run_command")
        dir_path = os.path.dirname(log_path)
        if not os.path.exists(dir_path) and dir_path != "":
            os.makedirs(dir_path)
        
        #檢查是否有儲存點
        max_epoch = -1
        max_resume = ""
        if(os.path.exists(output_dir)):
            for output_file in os.listdir(output_dir):
                output_filepath = os.path.join(output_dir, output_file)
                print(output_filepath)
                if(os.path.isdir(output_filepath)):
                    name, epoch, state = output_file.split("-")
                    epoch = int(epoch)
                    if (max_epoch < epoch):
                        max_epoch = epoch
                        max_resume = output_filepath
        print(f"max_epoch: {max_epoch}, max_resume: {max_resume}")
        kwargs["output_name"] = output_name
        kwargs["dataset_config"] = dataset_config
        kwargs["output_dir"] = output_dir
        cmd = get_command(max_epoch, max_resume, **kwargs)
        with open(log_path, "a", encoding="utf-8") as f:
            print(f"run_command: {cmd}")
            # process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f, text=True)
            process = subprocess.Popen(cmd, shell=True, text=True)
            process.wait()
        
        if process.returncode != 0:
            print(f"run_command error: {cmd}")
            print(f"run again")
            time.sleep(60)
        else:
            break

# if __name__ == "__main__":