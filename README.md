# maskfree_VTON

This project re-implements [OmniTry (Feng et al., 2025)](https://arxiv.org/abs/2508.13632) with a U-Net-based diffusion backbone and integrates techniques from [BooW-VTON (Li et al., 2024)](https://arxiv.org/abs/2408.06047) to improve in-the-wild performance.


Adjustment include:
- **Backbone Redesign:** Replaced the diffusion transformer backbone with a U-Net–based diffusion architecture.
- **Garment Transfer Mechanism:** Redesigned the garment transfer attention mechanism to operate without rotary embeddings and to work with Unet feature maps.
- **Data Augmentation:** Applied techniques from **BooW-VTON (Li et al., 2024)** to enhance in-the-wild robustness and generalization by compositing person images onto diverse backgrounds after removing the original background.
- Includes attention visualization.
- Includes evaluation script and the training script to train the localization & identiy LoRa. Tracks training progress using MLFlow.
