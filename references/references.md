# References & Scientific Foundation

This document details the scientific references, domain literature, and neural architecture choices supporting the FinFET Drift-Sense Localization System.

---

## 1. Semiconductor & SEM Imaging Domain Literature

1. **FinFET Structure & Critical Dimension Metrology**
   - *Reference*: IEEE Transactions on Semiconductor Manufacturing / Applied Materials SEM Metrology Standard.
   - *Summary*: Modern FinFET sub-7nm architecture features sub-30nm fin pitch and repeating gate structures. Microscopic spatial drift occurs due to mechanical stage drift, thermal expansion, and beam charging phenomena under Scanning Electron Microscopy (SEM).

2. **Scanning Electron Microscopy Image Formation & Noise Degradation**
   - *Reference*: Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*. Springer-Verlag.
   - *Summary*: SEM image degradation is characterized by Poisson shot noise from low primary electron dose ($200.0$ to $2000.0$), beam astigmatism, spatial drift jitter, and secondary electron emission variation across high-aspect-ratio trenches.

---

## 2. Deep Learning & Computer Vision Literature

3. **Siamese Neural Networks for One-Shot Visual Localization**
   - *Reference*: Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-shot Image Recognition*. ICML Deep Learning Workshop.
   - *Summary*: Shared-weight Siamese dual-branch architectures project disparate visual patches into a joint metric embedding space.

4. **MobileNetV3 Lightweight Backbone & SE Attention**
   - *Reference*: Howard, A., et al. (2019). *Searching for MobileNetV3*. IEEE/CVF International Conference on Computer Vision (ICCV).
   - *Summary*: Inverted bottleneck blocks with hard-swish activation and Squeeze-and-Excitation (SE/ECAM) channel attention achieve ultra-low parameter counts (~337K params) while maintaining feature representation power.

5. **Convolutional Block Attention Module (CBAM)**
   - *Reference*: Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module*. European Conference on Computer Vision (ECCV).
   - *Summary*: Sequentially combines channel attention (what features to focus on) and spatial attention (where structural features are located).

---

## 3. Explicit Differentiation of Components

| Component | Scientific Origin / Literature Basis | Our Proposed System Adaptation |
| :--- | :--- | :--- |
| **MobileNetV3 Backbone** | Howard et al. (2019) | Adapted stem layer to 1-channel $128 \times 128$ SEM grayscale inputs. |
| **CBAM Attention** | Woo et al. (2018) | Integrated after final inverted bottleneck to emphasize FinFET line edges. |
| **CIR Module** | Structural receptive field literature | Developed multi-branch receptive field (1x1, 3x3, 5x5) for SEM FinFET structures. |
| **Dual Correlation** | Cross-correlation literature | Designed explicit channel-wise + pixel-wise learnable correlation fusion. |
| **X/Y Feedback** | Coordinate regression literature | Formulated continuous sub-pixel offset regression heads ($\Delta x, \Delta y$). |
| **Hard-Negative Replay** | Experience Replay (Mnih et al.) | Engineered `ReplayMemory` prioritizing false-positive repeating FinFET patterns. |
| **Repeated-Pattern Rule** | Domain constraint | Applied center proximity tie-breaking rule at $(500, 500)$ for ambiguous candidates. |
