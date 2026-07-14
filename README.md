# VAE vs GAN on MNIST

Two generative models, same dataset, built in PyTorch — then compared on the two things that actually differ: sample quality and training stability.

## Structure

```
vae.py        # encoder → μ, σ → reparameterised sample → decoder
gan.py        # generator + discriminator, two optimizers, alternating updates
results.md    # the comparison: samples, loss curves, why
notes.md      # implementation notes
learning.md   # concept notes
plan.md
```

## Results

| | VAE | GAN |
|---|---|---|
| Samples | smoother, slightly blurred | sharper, more defined edges |
| Failure mode | none — every sampled `z` decodes to something legible | occasional non-digit garbage |
| Loss | monotonic (146 → 106 over 100 epochs) | oscillates (D: 0.90–1.32, G: 0.55–1.88) |

**Why VAE never outputs garbage:** the KL term regularises the *entire* latent space, so any sampled `z` lands somewhere the decoder learned as valid.

**Why GAN sometimes does:** G only gets gradient signal at noise vectors actually drawn during training — nothing constrains the rest of the space, so under-visited regions can decode to nonsense.

**The oscillation isn't failure.** VAE's stability is structural (encoder + decoder cooperate on one objective, one optimizer). GAN's instability is equally structural (two networks, opposing objectives, one shared landscape — every update to one changes what the other is optimising against).

Full write-up: [`results.md`](results.md)

## Stack

Python, PyTorch, MNIST via torchvision.
