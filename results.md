## Sample quality: VAE vs GAN
### VAE output
<img width="1179" height="295" alt="Screenshot 2026-07-14 at 17 31 10" src="https://github.com/user-attachments/assets/f6e9b9d6-b6e1-42a0-855f-e6834f06216a" />

### GAN output
<img width="1248" height="335" alt="Screenshot 2026-07-14 at 17 31 23" src="https://github.com/user-attachments/assets/2c8a13ef-cbfe-45ae-b85d-c3c36b65a8c7" />

- **Edges**: GAN fakes sharper/more defined; VAE reconstructions smoother, slightly blurred
- **Why VAE never outputs garbage**: KL term regularizes the *entire* latent space (not just regions near real images) → every sampled `z`, even from pure noise (`epsilon` in the reparam trick), lands somewhere the decoder learned as "valid"
- **Why GAN sometimes does**: G only gets gradient signal from noise vectors actually sampled during training — no constraint forcing the *whole* noise space to map to valid digits → under-visited regions can produce non-digit garbage (structurally impossible for VAE)
- **Consistency**: VAE reconstructions uniformly legible across all samples; GAN fakes more variable — mostly good, occasional unreadable outlier
- **Takeaway**: GAN trades consistency for sharper best-case quality; VAE trades peak sharpness for reliability


## Training stability: VAE vs GAN


### VAE loss (monotonic)
| Epoch | Total Loss | Rec Loss | KL Loss |
|---|---|---|---|
| 0  | 146.3777 | 131.4343 | 14.9434 |
| 10 | 111.5258 | 93.1984  | 18.3274 |
| 20 | 109.4257 | 90.8679  | 18.5578 |
| 30 | 108.4705 | 89.8110  | 18.6595 |
| 40 | 107.8112 | 89.1238  | 18.6875 |
| 50 | 107.4302 | 88.6961  | 18.7341 |
| 60 | 107.1097 | 88.3598  | 18.7499 |
| 70 | 106.9121 | 88.1634  | 18.7487 |
| 80 | 106.6750 | 87.9311  | 18.7439 |
| 90 | 106.4897 | 87.7378  | 18.7520 |
| 99 | 106.3915 | 87.6430  | 18.7485 |

<img width="661" height="382" alt="Screenshot 2026-07-14 at 17 50 36" src="https://github.com/user-attachments/assets/a7668e84-5dd9-4d60-a322-de7680f8196d" />

### GAN loss (oscillating)

| Epoch | D_loss | G_loss |
|---|---|---|
| 0  | 1.1201 | 1.0048 |
| 10 | 1.3218 | 0.5493 |
| 20 | 1.1424 | 1.1971 |
| 30 | 1.2966 | 1.0813 |
| 40 | 1.1554 | 1.2668 |
| 50 | 1.1264 | 1.1294 |
| 60 | 1.2184 | 1.2075 |
| 70 | 1.1017 | 1.3040 |
| 80 | 0.9159 | 1.3171 |
| 90 | 0.9009 | 1.8837 |
| 99 | 0.9974 | 1.3613 |

<img width="601" height="385" alt="Screenshot 2026-07-14 at 17 52 30" src="https://github.com/user-attachments/assets/83bca8fd-7d16-4c9f-83dc-7ad9fb71050a" />

- VAE loss is **monotonic**: 146 → 106 over 100 epochs, no epoch where loss 
  increases from the last. Single loss, single optimizer, encoder+decoder 
  cooperate on one objective — nothing fights the descent.
- GAN loss **oscillates**: D_loss and G_loss both move up and down throughout 
  training (D: 0.90–1.32, G: 0.55–1.88), never settling monotonically. Two 
  networks with opposing objectives share the same landscape — every update 
  to one changes what the other is optimizing against.
- This is **not collapse or failure** — no loss pinned near 0, samples stayed 
  legible throughout (see sample quality section). Oscillation here reflects 
  the adversarial minimax structure itself, not broken training.
- Takeaway: VAE's stability is structural (cooperative, single objective); 
  GAN's instability is also structural (adversarial, dual objective) — not a 
  sign either network failed, just an inherent property of each architecture.


**Latent space**: [your sentence above]

**GAN's advantage**: sharper, more realistic-looking outputs at the pixel level.
**GAN's cost**: unstable training (two competing losses vs one), no structured 
latent space, harder to debug when something goes wrong (loss numbers alone 
don't tell you if it's working — must eyeball samples).

