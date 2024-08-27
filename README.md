# Self Supervised Machine Learning For Predicting Cancer Dependencies

In this project, two new deep learning models, VAE-DeepDep and MAE-DeepDep, were developed to improve the performance of the existing DeepDEP model in predicting gene dependencies in cancer cells. VAE-DeepDep, based on variational autoencoders, emerged as the most successful model, followed by MAE-DeepDep, which utilizes masked autoencoders. Both models demonstrated superior performance compared to the original DeepDEP model. An innovative fine-tuning step, further enhanced model performance. The models were trained and validated using extensive datasets, including The Cancer Genome Atlas (TCGA), the Broad Institute’s Dependency Map (DepMap), and the Molecular Signature Database (MSigDB). The best model was used to calculate dependency scores of TCGA tumors on 1,298 selected genes. In variational autoencoders, the e↵ect of the   parameter, which controls the trade-o↵ between accurately reconstructing the input data and regularizing the latent space, was tested with four di↵erent   values, and the results were analyzed. It was found that varying   values did not lead to significant di↵erences in model performance. Furthermore, in dimensionality reduction tasks, the advantages and disadvantages of using di↵erent latent representations in variational autoencoders were investigated, but no significant performance di↵erences were observed between these representations. For masked autoencoders, the e↵ect of the mask ratio, which determines the proportion of input data that is masked during training, was tested using three di↵erent mask ratios. The results indicated that the optimal mask ratio was 0.75, but there was no significant di↵erence compared to the lowest mask ratio of 0.25. The study also investigated the e↵ects of variable learning rates on both VAE and MAE models and showed that extremely small and extremely large learning rates had a significant impact on performance. Paired t-test was performed to evaluate the performance di↵erence between VAE-DeepDep and DeepDEP models and it was revealed that VAE-DeepDep model performed better in terms of prediction success. Furthermore, an analysis using input dropout revealed that the fingerprint data contributed the most to model performance. Finally, synthetic lethality analyses of four genes were conducted using the MutFingerprint-VAE-DeepDep model, providing insights into potential therapeutic targets.

This repository contains the codes that provide the results and graphs shared in the dissertation. The repository opened at the beginning of the entire project can be found at this link: https://github.com/kemalbayikk/Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependincies
