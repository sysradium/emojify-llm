# ORPO-Based Fine-Tuning for Emoji Usage

## Introduction

This project implements various fine-tuning techniques to adapt an existing base model for generating responses that effectively incorporate emojis. One of the key techniques used is **ORPO (Odds Ratio Preference Optimization)**.

### What is ORPO?

Odds Ratio Preference Optimization (ORPO) is a fine-tuning approach designed to optimize models using reinforcement learning with preference data. It refines the model to align with human-like preferences by leveraging odds ratio-based preference modeling.

The reward function for ORPO, as per the original paper, is formulated as:

$$
R(y) = \log \frac{\pi_{\theta}(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \cdot A(y)
$$

where:

- $`\pi_{\theta}(y | x)`$ is the fine-tuned model's probability of generating response \( y \) given input \( x \),
- $`\pi_{\text{ref}}(y | x)`$ is the reference model's probability,
- $`A(y)`$ is the advantage function,
- $`\beta`$ is a scaling parameter controlling preference strength.

## Project Structure

The project consists of the following components:

### 1. `orpo/generate.py` - Generating a Preference Dataset

This script uses `distilabel` to generate a preference dataset. Given a prompt, it generates two responses using different models:

- One model is prompted to use emojis.
- The other model is prompted to avoid emojis.

The scoring function is currently simple:

$$
S(m) = \begin{cases} 1, & \text{if } m \text{ contains an emoji} \\ 0, & \text{otherwise} \end{cases}
$$

A more sophisticated scoring mechanism will be implemented later.

### 2. `orpo/train.py` - Training with ORPO and LoRA

This script fine-tunes the model using **Parameter-Efficient Fine-Tuning (PEFT)**, specifically **LoRA (Low-Rank Adaptation)**, implemented via **Hugging Face's ORPO framework**.

#### LoRA Overview

LoRA introduces low-rank trainable matrices to efficiently fine-tune pre-trained models:

$$
W' = W + \Delta W
$$

where $`W'`$ is the adapted weight matrix, and $`\Delta W`$ is decomposed as:

$$
\Delta W = A B
$$

with:

$$
A \in \mathbb{R}^{d \times r}, \quad B \in \mathbb{R}^{r \times k}
$$

This significantly reduces the number of trainable parameters while retaining model performance.

### 3. `orpo/inference.py` - Generating Text with the Fine-Tuned Model

This script loads the fine-tuned model and generates sample responses to demonstrate the effectiveness of ORPO training.

## Visualization

To monitor training progress, **TensorBoard** is used. It can be launched with:

```sh
make tensorboard
```

## Installation and Usage

1. Install dependencies using `uv`:

    ```sh
    uv sync
    ```

2. Generate preference data:

    ```sh
    $(PYTHON) orpo/generate.py
    ```

3. Train the model:

    ```sh
    $(PYTHON) orpo/train.py
    ```

4. Run inference:

    ```sh
    $(PYTHON) orpo/inference.py
    ```
