import json
import os
import subprocess
output_dir = "D:\AIGC\dataset\fun\LilyLinglan_model_xplusPoses_onlySD_datav2_part2"
output_name = "LilyLinglan"
max_train_epochs = 10

def get_command(initial_epoch, resume):
    if resume == "":
        keep_cmd = f'\
        accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py \
            --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\copaxTimelessxl_xplusPoses_onlySD.safetensors" \
            --clip_l="C:\ComfyUIModel\models\clip\clip_l.safetensors" \
            --t5xxl="C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors" \
            --ae="C:\ComfyUIModel\models\vae\ae.safetensors" \
            --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module networks.lora_flux --network_dim 32 --optimizer_type adamw8bit --learning_rate 5e-5 \
            --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs=1 --dataset_config="../config_lin.toml" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --timestep_sampling=sigmoid --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
            --initial_epoch={initial_epoch} --skip_until_initial_step \
            --resume="{resume}" \
            --lowram --save_state '
    else:
        keep_cmd = f'\
        accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py \
            --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\copaxTimelessxl_xplusPoses_onlySD.safetensors" \
            --clip_l="C:\ComfyUIModel\models\clip\clip_l.safetensors" \
            --t5xxl="C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors" \
            --ae="C:\ComfyUIModel\models\vae\ae.safetensors" \
            --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
            --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
            --network_module networks.lora_flux --network_dim 32 --optimizer_type adamw8bit --learning_rate 5e-5 \
            --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
            --highvram --max_train_epochs {max_train_epochs} --save_every_n_epochs=1 --dataset_config="../config_lin.toml" \
            --output_dir="{output_dir}" \
            --output_name="{output_name}" \
            --timestep_sampling=sigmoid --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
            --lowram --save_state '
    return keep_cmd

log_path = os.path.join(output_dir, "train.log")
while(True):
    print("run_command")
    dir_path = os.path.dirname(log_path)
    if not os.path.exists(dir_path) and dir_path != "":
        os.makedirs(dir_path)
    
    #檢查是否有儲存點
    for output_file in os.listdir(output_dir):
        max_epoch = 0
        max_resume = ""
        if(os.path.isdir(output_file)):
            name, epoch, state = output_file.split("_")
            epoch = int(epoch)
            if (max_epoch < epoch):
                max_epoch = epoch
                max_resume = os.path.join(output_dir, output_file)

    cmd = get_command(max_epoch, max_resume)
    with open(log_path, "a") as f:
        process = subprocess.Popen(cmd, shell=True, text=True)
        process.wait()
    
    if process.returncode != 0:
        print(f"run_command error: {cmd}")
        print(f"run again")
    else:
        break