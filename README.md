Jigsaw - Agile Community Rules Classification

This repo contains my work for the Jigsaw Agile Community Rules Kaggle competition.
The task is to classify Reddit comments based on whether they violate the community rules or not.

I used LLMs ranging from 1B to 32B, both as-is and with LoRA finetuning using PEFT and Unsloth libraries.
I also built a vector-similarity–based classifier using Faiss to support the prediction workflow.

This repository includes the training scripts, inference pipeline, and utilities used during the competition.
