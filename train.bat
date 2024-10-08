call .\venv\Scripts\activate
cd ./sd-scripts
@REM accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py ^
@REM     --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\flux1-dev.safetensors"  ^
@REM     --clip_l="C:\ComfyUIModel\models\clip\clip_l.safetensors"  ^
@REM     --t5xxl="C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors" ^
@REM     --ae="C:\ComfyUIModel\models\vae\\ae.safetensors" ^
@REM     --cache_latents_to_disk --cache_latents --save_model_as safetensors  ^
@REM     --sdpa --persistent_data_loader_workers  ^
@REM     --max_data_loader_n_workers 2 --gradient_checkpointing ^
@REM     --mixed_precision bf16 --save_precision bf16 --network_module networks.lora_flux  ^
@REM     --network_dim 16 --optimizer_type adamw8bit --learning_rate 1e-4  ^
@REM     --network_train_unet_only --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk  ^
@REM     --fp8_base --highvram --max_train_epochs 16 --save_every_n_epochs 4 --dataset_config="../config.toml"  ^
@REM     --output_dir="D:\AIGC\model\stegosaurus_full"  ^
@REM     --output_name="stegosaurus_flux-lora"  ^
@REM     --timestep_sampling sigmoid --model_prediction_type raw --guidance_scale 1.0 --loss_type l2 ^
@REM     --log_with wandb --wandb_run_name flux-lora-train

@REM accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py ^
@REM     --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\copaxTimelessxl_xplusPoses_onlySD.safetensors" ^
@REM     --clip_l="C:\ComfyUIModel\models\clip\clip_l.safetensors" ^
@REM     --t5xxl="C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors" ^
@REM     --ae="C:\ComfyUIModel\models\vae\ae.safetensors" ^
@REM     --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers ^
@REM     --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 ^
@REM     --network_module networks.lora_flux --network_dim 32 --optimizer_type adamw8bit --learning_rate 5e-5 ^
@REM     --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base ^
@REM     --highvram --max_train_epochs 10 --save_every_n_epochs 1 --dataset_config="../config_lin.toml" ^
@REM     --output_dir="D:\AIGC\dataset\fun\LilyLinglan_model_xplusPoses_onlySD_datav2_part2" ^
@REM     --output_name="LilyLinglan" ^
@REM     --sample_prompts="D:\AIGC\model\oneImageTest\sample_prompts.json" ^
@REM     --timestep_sampling=sigmoid --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 ^
@REM     --log_with wandb --wandb_run_name flux-lora-train ^
@REM     --lowram --save_state


accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py ^
    --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\copaxTimelessxl_xplusPoses_onlySD.safetensors" ^
    --clip_l="C:\ComfyUIModel\models\clip\clip_l.safetensors" ^
    --t5xxl="C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors" ^
    --ae="C:\ComfyUIModel\models\vae\ae.safetensors" ^
    --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers ^
    --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 ^
    --network_module networks.lora_flux --network_dim 32 --optimizer_type adamw8bit --learning_rate 5e-5 ^
    --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base ^
    --highvram --max_train_epochs 10 --save_every_n_epochs=1 --dataset_config="../config_lin.toml" ^
    --output_dir="D:\AIGC\dataset\fun\LilyLinglan_model_xplusPoses_onlySD_datav2_part2" ^
    --output_name="LilyLinglan" ^
    --sample_prompts="D:\AIGC\model\oneImageTest\sample_prompts.json" ^
    --timestep_sampling=sigmoid --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 ^
    --initial_epoch=4 --skip_until_initial_step ^
    --resume="D:\AIGC\dataset\fun\LilyLinglan_model_xplusPoses_onlySD_datav2_part2\LilyLinglan-000003-state" ^
    --lowram --save_state

cd ..


@REM --log_with wandb --wandb_run_name flux-lora-train ^

@REM --resume="D:\AIGC\dataset\fun\LilyLinglan_model_xplusPoses_onlySD_datav2_part2" ^