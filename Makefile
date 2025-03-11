.PHONY: train_orpo tensorboard

train_orpo:
	DATASET="Orion-zhen/dpo-mathinstuct-emoji" BASE_MODEL="Qwen/Qwen2-0.5B-Instruct" uv run python orpo/train.py

tensorboard:
	uv run tensorboard --logdir=./trained --port 6006
