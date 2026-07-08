# Notes 
## Autoencoder → VAE
### [variational autoencoders by arxiv insights](https://www.youtube.com/watch?v=9zKuYvjFFS8)

### [Intuitively Understanding Variational Autoencoders](https://medium.com/data-science/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)


**Why VAEs?**

Generative models can make brand-new random outputs, but often you want to **explore/alter existing data** in a specific direction — VAEs are especially good at this.

**Standard Autoencoder**

- Two networks: **encoder** (input → small dense encoding) + **decoder** (encoding → reconstruct input)
- Trained via **reconstruction loss** (MSE or cross-entropy between output and input)
- Encoder is forced to discard unimportant info, keep only what's needed to reconstruct

**The problem**

- The latent space learned this way is **not continuous** — it forms disconnected clusters
- Fine for reconstructing known inputs, but bad for generation: sampling a random point between clusters → decoder has never seen that region → garbage output

**What VAEs change**
- Encoder outputs **two vectors** instead of one encoding: a mean vector **μ** and a std-dev vector **σ**
- These define a distribution per input; the actual encoding is **sampled** from N(μ, σ²)
- Same input → slightly different encoding every pass (stochastic)
- Intuition: μ = where the encoding is centered, σ = how much it's allowed to vary
- Because decoder sees many sampled variations of the same input during training, it learns nearby points in latent space also decode sensibly → smooths the space **locally**

$$
\mathcal{L}_{VAE} = \mathcal{L}_{recon} + \mathcal{L}_{KL}
$$

purely **reconstruction loss**: the same as that used in MLP, can be Binary Cross-Entropy, Mean Squared Error, etc. 

<img width="687" height="355" alt="Screenshot 2026-07-08 at 13 51 43" src="https://github.com/user-attachments/assets/ae7aeb23-056f-4cb1-8851-4f29e8c1c42c" />

purely **KL loss**: Pulls all encodings toward the center, densely packed, evenly spread

- Add KL divergence term to the loss: penalizes how far the learned distribution (μ, σ) is from a standard normal N(0, I)

<img width="674" height="659" alt="Screenshot 2026-07-08 at 13 52 14" src="https://github.com/user-attachments/assets/bc34199a-c39d-4ee7-92d3-f0c704af3c1a" />


**Both together**:

<img width="673" height="666" alt="Screenshot 2026-07-08 at 13 53 46" src="https://github.com/user-attachments/assets/54c231d0-5281-4436-b2a2-f21593134c05" />

