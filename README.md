# Self-Supervised Handwritten Character Recognition Using Generative Adversarial Networks

This repository contains the code for the implementation of supervised HCR, character generation, and self-supervised HCR as described in the Thesis "Towards Self-Supervised Handwritten Text Recognition using Generative Adversarial Networks".

The purpose of this was to explore the feasbility of the self-supervised HTR framework proposed in the thesis on a more simple problem, and to investigate how it can be implemented. 

To do this, it was first ascertained that a decent performing supervised HCR model could be obtained s.t. the set up could be used for self-supervised HCR training. Next, a generator was obtained by training a conditional GAN adapted from Keras. Lastly, the pretrained generator was used in the self-supervised framework to produce synthetic images with the predicted digits that were compared to the input images. All experiments were conducted on MNIST.

The key component in the novel image-based self-supervised framework is the loss function, which needs to be able to compare two images based on their content. We experimented with three loss functions based on image generation literature:
- Mean Squared Error
- Binary Cross-Entropy
- Perceptual loss

Files mnist_gan.ipynb and gan_model.py contain the implementation of the DCGAN (adapted from https://keras.io/examples/generative/conditional_gan/) \
Files HTR_mwe.ipynb and self_htr.py contain the implementation of the supervised and self-supervised HCR models. 
