.PHONY: train_orpo tensorboard

PYTHON ?= uv run

train_orpo:
	DATASET="Orion-zhen/dpo-mathinstuct-emoji" BASE_MODEL="Qwen/Qwen2-0.5B-Instruct" $(PYTHON) accelerate launch orpo/train.py --use-peft

tensorboard:
	$(PYTHON) tensorboard --logdir=./trained --port 6006

mlflow:
	$(PYTHON) mlflow ui --host 127.0.0.1 --port 5000
