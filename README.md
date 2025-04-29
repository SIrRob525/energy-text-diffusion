# Diffusion Models for Text Generation
Reproducing results of "Xu et al. (2025). Energy-Based Diffusion Language Models for Text Generation. https://arxiv.org/abs/2410.21357"

## Description

This repository contains an implementation of a masked diffusion model for text generation based on the EDLM paper.

File structure:
- dataset.py: load passages from OpenWebText
- model.py: classic masked diffusion (forward, backward, sampling, generation)
- EDLM.py: EDLM implementaition (compute energy, new sampling)
- train.py: training loop
- demo.ipynb: demo notebook to generate some text