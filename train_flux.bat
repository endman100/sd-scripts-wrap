call .\venv\Scripts\activate
cd ./sd-scripts
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train.py   ^
    --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\flux1-dev.safetensors" ^
    --clip_l="C:\ComfyUIModel\models\clip\clip_l.safetensors" ^
    --t5xxl="C:\ComfyUIModel\models\clip\t5xxl_fp16.safetensors" ^
    --ae="C:\ComfyUIModel\models\vae\\ae.safetensors" ^
    --save_model_as safetensors --sdpa --persistent_data_loader_workers --max_data_loader_n_workers 1 ^
    --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 ^
    --dataset_config="../config.toml" ^
    --output_dir="D:\AIGC\model\stegosaurus_flux_single" ^
    --output_name="output-name" ^
    --learning_rate 1e-4 --max_train_epochs 3  --sdpa --highvram --cache_text_encoder_outputs_to_disk --cache_latents_to_disk --save_every_n_epochs 3 ^
    --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" ^
    --lr_scheduler constant_with_warmup --max_grad_norm 0.0 ^
    --timestep_sampling="shift" --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 ^
    --fused_backward_pass  --double_blocks_to_swap 6 --cpu_offload_checkpointing --full_bf16

cd ..