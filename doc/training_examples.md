# Training Examples

The integrated cell code base supports training of many types of models. 
We use a simple taxonomy to identify how to load different network components:

Autoencoders (`ae`) contain an encoder and a decoder. Examples of this are:  
- vanilla autoencoders
- [variational autoencoders](https://arxiv.org/abs/1312.6114)
- [conditional variational autoencoders](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)

Advarsarial Autoencoders (`aae`) contain an advarsary on the encoder in addition to the decoder.
- [advarsarial autoencoders](https://arxiv.org/abs/1511.05644)

Autoencoder GANs (`aegan`) contain a GAN attached to the decoder.
- [Autoencoder GANs](https://arxiv.org/abs/1512.09300)

Advarsarial Autoencoder GANs (`aaegan`) contain a advarsary attached to the encoder and a different advarsary attached to the decoder.

Examples of how to configure training code for different models is as follows:


