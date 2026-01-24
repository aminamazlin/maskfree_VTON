# maskfree_VTON

This project re-implements [OmniTry (Zhou et al., 2023)](https://arXiv:2508.13632) with a U-Net-based diffusion backbone and integrates techniques from [BooW-VTON (Author et al., 2022)](https://arXiv:2408.06047) to improve in-the-wild performance.


Key contributions include:
- **Backbone Redesign:** Replaced the diffusion transformer backbone with a U-Net–based diffusion architecture
- **Garment Transfer Mechanism:** Redesigned the garment transfer attention mechanism to operate without rotary embeddings and to work with Unet feature maps.
- **Data Augmentation:** Applied techniques from **BooW-VTON** to enhance in-the-wild robustness and generalization by compositing person images onto diverse backgrounds after removing the original background.
- Includes the training script to train the localization & identiy LoRa and tracks training progress using MLFlow
