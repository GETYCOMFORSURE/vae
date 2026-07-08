# Notes

## Autoencoder → VAE

### Resources
- [Variational Autoencoders by Arxiv Insights](https://www.youtube.com/watch?v=9zKuYvjFFS8)
- [Intuitively Understanding Variational Autoencoders](https://medium.com/data-science/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

### Why Variational Autoencoders (VAEs)?
Generative models can make brand-new random outputs, but often you want to **explore/alter existing data** in a specific direction — VAEs are especially good at this.

### Standard Autoencoder
- Two networks: **encoder** (input → small dense encoding) + **decoder** (encoding → reconstruct input)
- Trained via **reconstruction loss** (MSE or cross-entropy between output and input)
- Encoder is forced to discard unimportant info, keep only what's needed to reconstruct

The problem

- The latent space learned this way is **not continuous** — it forms disconnected clusters
- Fine for reconstructing known inputs, but bad for generation: sampling a random point between clusters → decoder has never seen that region → garbage output

### What VAEs change
- Encoder outputs **two vectors** instead of one encoding: a mean vector **μ** and a std-dev vector **σ**
- These define a distribution per input; the actual encoding is **sampled** from N(μ, σ²)
- Same input → slightly different encoding every pass (stochastic)
- Intuition: μ = where the encoding is centered, σ = how much it's allowed to vary
- Because decoder sees many sampled variations of the same input during training, it learns nearby points in latent space also decode sensibly → smooths the space **locally**

### Loss Function

$$
\mathcal{L}_{VAE} = \mathcal{L}_{recon} + \mathcal{L}_{KL}
$$

#### Reconstruction loss
- the same as that used in MLP, can be Binary Cross-Entropy, Mean Squared Error, etc.
- purely **reconstruction loss** (see below)

<img width="687" height="355" alt="Screenshot 2026-07-08 at 13 51 43" src="https://github.com/user-attachments/assets/ae7aeb23-056f-4cb1-8851-4f29e8c1c42c" />

#### KL loss
- Add KL divergence term to the loss: penalizes how far the learned distribution (μ, σ) is from a standard normal N(0, I)
- purely **KL loss** (see below): Pulls all encodings toward the center, densely packed, evenly spread


<img width="674" height="659" alt="Screenshot 2026-07-08 at 13 52 14" src="https://github.com/user-attachments/assets/bc34199a-c39d-4ee7-92d3-f0c704af3c1a" />

#### Both together

<img width="673" height="666" alt="Screenshot 2026-07-08 at 13 53 46" src="https://github.com/user-attachments/assets/54c231d0-5281-4436-b2a2-f21593134c05" />

### Code

#### Encoding section: encoder + reparameterization trick

```python
# build your encoder upto here. It can simply be a series of dense layers, a convolutional network
# or even an LSTM decoder. Once made, flatten out the final layer of the encoder, call it hidden.

# we use Keras to build the graph

latent_size = 5
mean = Dense(latent_size)(hidden)

# we usually don't directly compute the stddev σ 
# but the log of the stddev instead, which is log(σ)
# the reasoning is similar to why we use softmax, instead of directly outputting
# numbers in fixed range [0, 1], the network can output a wider range of numbers which we can later compress down
log_stddev = Dense(latent_size)(hidden)

def sampler(mean, log_stddev):
    # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    std_norm = K.random_normal(shape=(K.shape(mean)[0], latent_size), mean=0, stddev=1)
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    return mean + K.exp(log_stddev) * std_norm
  
latent_vector = Lambda(sampler)([mean, log_stddev])
# pass latent_vector as input to decoder layers
```

#### Loss section

```python
def vae_loss(input_img, output):
    # compute the average MSE error, then scale it up i.e. simply sum on all axes
    reconstruction_loss = K.sum(K.square(output-input_img))
    # compute the KL loss
    kl_loss = -0.5 * K.sum(1 + log_stddev - K.square(mean) - K.square(K.exp(log_stddev)), axis=-1)
    # return the average loss over all images in batch
    total_loss = K.mean(reconstruction_loss + kl_loss)
    return total_loss
```
