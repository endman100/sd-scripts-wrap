call .\venv\Scripts\activate
cd ./sd-scripts
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py ^
    --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\flux1-dev.safetensors"  ^
    --clip_l="C:\ComfyUIModel\models\clip\clip_l.safetensors"  ^
    --t5xxl="C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors" ^
    --ae="C:\ComfyUIModel\models\vae\\ae.safetensors" ^
    --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers ^
    --max_data_loader_n_workers 1 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 ^
    --network_module networks.lora_flux --network_dim 32 --optimizer_type adamw8bit --learning_rate 5e-5 ^
    --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base ^
    --highvram --max_train_epochs 2 --save_every_n_epochs=2 --dataset_config="../config_one.toml" ^
    --train_data_dir="D:\AIGC\dataset\oneImageTest" ^
    --output_dir="D:\AIGC\dataset\oneImageTest" ^
    --output_name="M1" ^
    --timestep_sampling="faster" --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 ^
    --lowram ^
    --log_with wandb --wandb_run_name="oneImageTest" --log_tracker_name="FxGame AIGC Tools flux"
cd ..