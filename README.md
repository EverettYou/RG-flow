# RG-Flow (ODE implementation)
 
This is a Pytorch implementation of the arXiv paper [arXiv:2203.07975](https://arxiv.org/abs/2203.07975): Artan Sheshmani, Yizhuang You, Wenbo Fu, and Ahmadreza Azizi, Categorical Representation Learning and RG flow operators for algorithmic classifiers.
 
 **RG-Flow** is a hierarchical flow-based generative model built on the idea of renormalization group (RG) in physics. It was originally introduced in Ref. [1] under the name of "NeuralRG" as a flow-based generative model on a multi-scale entanglement renormalization ansatz (MERA) network structure in physics. Ref. [2] lays down the theoretical foundation between the hierarchical flow-based generative model and the modern understanding of renormalization group flow as an optimal transport that disentangles a quantum field theory. The architecture is simplified as the model develops. The technology is further applied to image generation [3] and sequence generation [4]. This repository hosts an implementation of RG-Flow based on neural ODE bijectors. It can learn to generate new samples and estimate sample log-likelihood given (i) either a set of training samples (ii) or an energy function that describes the sample distribution (as a Boltzmann distribution).

[1] [arXiv:1802.02840](https://arxiv.org/abs/1802.02840): Shuo-Hui Li and Lei Wang, *Neural Network Renormalization Group*. Associtated GitHub repository: [NeuralRG](https://github.com/li012589/NeuralRG)

[2] [arXiv:1903.00804
](https://arxiv.org/abs/1903.00804): Hong-Ye Hu, Shuo-Hui Li, Lei Wang, Yi-Zhuang You. *Machine Learning Holographic Mapping by Neural Network Renormalization Group*.

[3] [arXiv:2010.00029](https://arxiv.org/abs/2010.00029): Hong-Ye Hu, Dian Wu, Yi-Zhuang You, Bruno Olshausen, Yubei Chen. *RG-Flow: A hierarchical and explainable flow model based on renormalization group and sparse prior*. Associtated GitHub repository: [RG-Flow (MERA implementation)](https://github.com/hongyehu/RG-Flow)

[4] [arXiv:2203.07975](https://arxiv.org/abs/2203.07975): Artan Sheshmani, Yizhuang You, Wenbo Fu, and Ahmadreza Azizi. *Categorical Representation Learning and RG flow operators for algorithmic classifiers*. 
