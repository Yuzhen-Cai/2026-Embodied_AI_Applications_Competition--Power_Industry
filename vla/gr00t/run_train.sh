
CUDA_VISIBLE_DEVICES=0 uv run python     gr00t/experiment/launch_finetune.py  --use_wandb   --base-model-path ./GR00T-N1.7-3B     --dataset-path demo_data/cube_to_bowl_5     --embodiment-tag NEW_EMBODIMENT     --modality-config-path examples/SO100/so100_config.py     --num-gpus 1     --output-dir save_finetune/test_finetune2     --max-steps 2000     --global-batch-size 32     --dataloader-num-workers 4


