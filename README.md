# Diffusion Models for Text Generation
Reproducing results of "Xu et al. (2025). Energy-Based Diffusion Language Models for Text Generation. https://arxiv.org/abs/2410.21357"

## Description

This repository contains an implementation of a masked diffusion model for text generation based on the EDLM paper.

- **dataset.py**: WikiText-103 dataset and dataloader implementation. Run this file to view sample text passages from the dataset.
- **model.py**: Implementation of a basic masked diffusion model. Run this file to see the forward process (masking), backward process (prediction), and generation process of the (untrained) model.
- **train.py**: TODO