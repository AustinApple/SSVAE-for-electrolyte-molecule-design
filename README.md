# SSVAE for electrolyte molecule design
The repository includes the semi-supervised VAE model and the execution script for automatical electrolyte molecular generation. 
The code is mainly forked by https://arxiv.org/abs/1805.00108. The method allows us to generate molecules with the desire property. 
In our work, molecules with IE and EA properties (from Material Project https://materialsproject.org/) and molecules without 
properties (from zinc) are used to train the SSVAE model. The architecture of the SSVAE model is shown in figure below.
![](image/architecture/SSVAE.pdf)
The semi-supervised VAE (SSVAE) is mainly composed of a proprety predictor and a variational autoencoder. A property prediction is
responsible for predicting 
