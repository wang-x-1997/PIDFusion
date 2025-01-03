# PIDFusion
Source code of the paper ***"PID Controller-Driven Network for Image Fusion"*** which has been accepted by TMM.

# Abstract
With its well-designed network architecture, the deep learning-based infrared and visible image fusion (IVIF) method shows its efficiency and effectiveness by realizing a fine feature extraction and fusion mechanism. However, disparities in cross-modal features often result in an imbalance between texture details and contextual information, causing detailed features to be overshadowed by prevailing contextual information. To tackle this issue, this study introduces PIDFusion, a fusion model driven by a PID controller, designed to dynamically optimize cross-modal feature fusion deviations. The core of PIDFusion is the dynamic adaptation capability of the PID controller, which facilitates real-time corrections for deviations encountered during the fusion process, thereby maintaining a harmonious balance between texture details and contextual information. 
Additionally, we introduced the Cyclic Self-Supervised Feature Refinement (CSSFR), which under the constraint of self-supervised loss, minimizes redundant information within the feature flow and ensures the preservation of salient feature through the cyclic input of decoupled features. Concurrently, we developed the Iterative Attention Module (IAM), utilizing the unique gating mechanism of LSTM to capture feature changes across successive iterations, thereby driving the model to cultivate more discriminative feature representations. Extensive experiments revealed that PIDFusion outperforms SOTA methods in terms of both efficiency and cost-effectiveness, through static statistics and high-level vision tasks. 
# :triangular_flag_on_post: Illustration of our PIDFusion
![The framework of PIDFusion](RDMFuse_img/RDMFuse.png)

# :triangular_flag_on_post: Testing
If you want to infer with our PIDFusion and obtain the fusion results in our paper, please run ```test.py```.
Then, the fused results will be saved in the ```'./Fused image/'``` folder.

# :triangular_flag_on_post: Training
You can change your own data address in ```dataset.py``` and use ```train.py``` to retrain the method.




## 🚀 Related Work
- Xue Wang, Zheng Guan, Wenhua Qian, Jinde Cao, Shu Liang, Jin Yan. *CS²Fusion: Contrastive learning for Self-Supervised infrared and visible image fusion by estimating feature compensation map*. **INF FUS 2024**, [https://www.sciencedirect.com/science/article/abs/pii/S156625352300355X](https://www.sciencedirect.com/science/article/abs/pii/S156625352300355X)
- Xue Wang, Zheng Guan, Wenhua Qian, Jinde Cao, Chengchao Wang, Runzhuo Ma. *STFuse: Infrared and Visible Image Fusion via Semisupervised Transfer Learning*. **TNNLS 2024**, [https://ieeexplore.ieee.org/abstract/document/10312808](https://ieeexplore.ieee.org/abstract/document/10312808)
- Xue Wang, Zheng Guan, Wenhua Qian, Jinde Cao, Chengchao Wang, Chao Yang.  *Contrast saliency information guided infrared and visible image fusion*. **TCI 2023**, [https://ieeexplore.ieee.org/abstract/document/10223277](https://ieeexplore.ieee.org/abstract/document/10223277)
- Xue Wang, Zheng Guan, Shishuang Yu, Jinde Cao, Ya Li. *Infrared and visible image fusion via decoupling network*. **TIM 2022**, [https://ieeexplore.ieee.org/abstract/document/9945905](https://ieeexplore.ieee.org/abstract/document/9945905)
- Zheng Guan, Xue Wang, Rencan Nie, Shishuang Yu, Chengchao Wang. *NCDCN: multi-focus image fusion via nest connection and dilated convolution network*. **Appl Intel 2022**, [https://link.springer.com/article/10.1007/s10489-022-03194-z](https://link.springer.com/article/10.1007/s10489-022-03194-z)


