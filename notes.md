# Notes
some notes jotted down during coding

## divde the components
### components in any ML workflow (near-universal, especially in PyTorch)
1. Data loader — get data in, clean, ready to feed
2. Model — define the architecture (what transforms input → output)
3. Loss function — define what "wrong" means, numerically
4. Training loop — repeatedly: feed batch → forward pass → compute loss → backward pass → update weights

(optional, add after the above 4 are working)

5. Evaluation — check performance on held-out data / sanity-check outputs (e.g. do generated digits actually look real, not just "loss went down")
6. Inference/deployment — use the trained model on new data, only needed if packaging for others to use


in other words: 
- workflow: encoder → sampler → decoder → compute loss → optimizer updates weights → repeat for next batch
- one pass through (encoder→sampler→decoder→loss→optimizer update) = one training step (or one batch/iteration)

## 1. vae - data loader

### recall: method, function, attribute, argument
- method - a type of function
- attribute - characters that make up the object
- object - can be lists, strings, etc.
```python
object.attribute # eg. print(dog1.name)
list[attribute]

object.method(argument)
list.method(argument)
```
### recall: save changes in github codespace using terminal
```python
git add .
git commit -m "your commit message here"
git push
```

### components in any data loader (in order)
- Get raw data into memory (**load**)
- Separate data you train on from data you evaluate on (**split**) — do this before normalizing, so you don't leak test-set information into your normalization stats
- Make sure values are in a range the model can learn from efficiently (**normalize**) — because gradient descent behaves badly on wildly different feature scales; compute mean/std from the training set only
- Randomize order (**shuffle**) — because if data comes in a meaningful order (e.g. sorted by label), the model can learn spurious patterns from the order itself instead of the actual features
- Feed data to the model in small groups, not all at once (**batch**) — memory can't hold the whole dataset at once, and the model updates its weights after each small batch (not once per epoch), giving many fast, stable updates instead of one huge slow one

### explain data loader decisions
- [MNIST documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
```python
data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform.ToTensor())
```
- `root='./data'` — this creates a folder called data
- `train=True` — True and False corresponds to two datasets with 60k and 10k images, respectively. I use more data (60k) to train, and use less (10k) to test.
- `download=True` — puts it in root directory (data folder)
- `transform=transforms.ToTensor()` — the data loaded is in PIL (python imaging library), but pytorch only works with tensor, so have to transform to tensor
  - what does ToTensor do?
    1. Reshapes (from PIL's (Height, Width, Channels) to PyTorch's (Channels, Height, Width))
    2. Rescales (from 0-255 integer range down to 0.0-1.0 floats) — not need to normalize anymore

### data structure returned by the dataset
- indexing returns a tuple: `(image, target)` according to the documentation
- **target** = label = the correct answer the model should predict (the digit 0–9). This is needed for supervised learning. But VAE is unsupervised, so just discard target in the training loop by (`for images, _ in dataloader`)

```python
print(data[0]) # first `(image, target)` pair
print(data[0][0]) # just the image
print(data[0][1]) # just the label
```

- to take a glimpse of the data:
```python
image, label = data[0]
plt.imshow(image.squeeze(), cmap='gray')
# `.squeeze()` — drops the size-1 channel dim: `[1,28,28]` → `[28,28]`, since imshow needs 2D
# `imshow(...)` — plots a 2D array as an image
# `cmap='gray'` — grayscale colormap; without it, matplotlib defaults to a colored map (viridis), misrepresenting a B&W digit
plt.title(f'Label: {label}')
plt.show()
```
### take a glimpse of items in DataLoader
- [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/stable/data.html): DataLoader is **iterable**, not indexable
- source shows internally it's built with `yield` (it hands you one batch at a time, only when you ask — via a `for` loop, or via `next(iter(...))`) — confirms: get items by iterating, not indexing
```python
images, labels = next(iter(train_loader))  # grabs the first batch
image = images[0]   # tensor IS indexable, unlike the loader
label = labels[0]
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()
```
- `iter(train_loader)` — turns it into an iterator
- `next(...)` — pulls the first yielded batch

## 2. vae - model
### encoder
-> basically a MLP but output two outputs (mean + standard deviation)

reference source code:
```python
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

resources:
- [a model with multiple outputs](https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440)
- [module documentation](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html)

### fun fact (my own opinion)
- Senior engineer = fast pattern-matching across many solved problems (horizontal).
- Senior researcher = knowing exactly where a deep field's unsolved edge sits and pushing past it (vertical).

### sampler

**where does it go — before or after μ, σ?**
- sampler = draw one random point near μ, spread by σ. `z = μ + σ·ε`, `ε ~ N(0,1)`
- so sampler goes after

**what's latent dimension**
- = number of the degree of freedom of one image -> label, thickness, style, etc -> so 1 dimension can't hold all of it
- no need to hand-assign what each dimension represents, because this is unsupervised learning (network finds its own factorization by minimizing loss)
- picked: latent_dim = 10

### decoder
- initial guess: decoder is roughly the inverse of encoder

**does decoder output distribution like encoder?**
- no — one number per pixel (0-1 brightness), not two
- encoder needs distribution because latent space must be smooth/continuous for generation
- decoder's job: map one z → one image, no smoothness requirement on pixel output

**how many layers**
- two (matches encoder's two layers before branching): `10 → 200 → 784`

**sigmoid at the end — why**
- MNIST pixels normalized to 0-1
- plain `Linear` can output any real number
- sigmoid squashes to [0,1] — not for "binary," for range-matching

## 1. gan - data loader
normalization is needed in the transform. GANs conventionally use Tanh as the generator's final activation instead of sigmoid, so VAE will normalize in sigmoid step whereas GAN has to have normalization separately.


## 3. loss + optimizer

### loss: BCE + the three targets
```python
criterion = nn.BCELoss()
```
- D ends in sigmoid, judging real vs fake → binary → BCE. Same pairing as the cancer MLP.
- No `reduction='sum'` — that only matters when adding two losses of different scale (VAE's BCE + KL). One term here, default `mean` is fine.

### two optimizers, not one
| phase | input to D | target | updates |
|---|---|---|---|
| train D | real | **1** | D |
| train D | fake | **0** | D |
| train G | fake | **1** | G |

- Same fake image, label **0** for D, label **1** for G. That lie *is* the adversarial game.
- Why a separate G phase? D's phase gives G no gradient path. Phase 2 exists solely to give G a signal, and the only one available is "make D say real."

```python
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
generator_optim     = torch.optim.Adam(generator.parameters(),     lr=0.0002, betas=(0.5, 0.999))
```
- G and D want **opposite** things. One optimizer descending one loss over both parameter sets is incoherent — nothing learns.
- VAE could use one optimizer because encoder + decoder *cooperate* on a shared goal. **This is the structural difference between GAN and everything before it.**

### what Adam is, and the GAN overrides
- Optimizer = the rule turning gradients into weight updates. Plain SGD = what I wrote by hand in NumPy: `W1 = W1 - lr * dW1`.
- Adam adds **momentum** (running average of past gradients → less zigzag) and **per-parameter learning rates** (scale each step by that weight's typical gradient size). Fast, low-tuning, default everywhere.
- Alternatives: SGD (plain, slow, sometimes generalises better) · SGD+momentum · RMSprop (Adam's ancestor, old GAN default) · AdamW (transformers/LLMs).
- Defaults: `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)`
- **GAN overrides:** `lr=0.0002` (0.001 destabilises) and `betas=(0.5, 0.999)` — lower first beta = less momentum. Momentum assumes a stable loss landscape; in a GAN your opponent is also learning, so the landscape moves under you.

## 4. training loop

### the two nested loops
- 60000 ÷ 32 = 1875 **batches**; one full pass over all of them = **1 epoch**.
- **Why 100 epochs on the same data?** Each pass = ~1875 tiny nudges per weight. A net doesn't learn from one exposure any more than I learn a paper from one read. Each pass starts from better weights than the last.
- `for images, _ in train_loader` — `_` discards MNIST's digit labels (GAN is unsupervised).
- **`.view(images.size(0), -1)`** flattens `[32,1,28,28]` → `[32,784]` for `nn.Linear(784,…)`. Same numbers, new shape.
  - `.size(0)` reads the batch size off the tensor — don't hardcode 32, the last batch may be smaller.
  - **`-1` = "solve for this."** `.view()` is a constraint equation: total elements fixed, one unknown → solvable. Only one `-1` allowed.

### the four-line rhythm — universal, not a GAN thing
`zero_grad → loss → backward → step`
- **`zero_grad()`** — PyTorch *accumulates* gradients; every `backward()` **adds** to what's stored. Without wiping, batch 2's gradient = batch1 + batch2, forever.
- **`backward()`** — autograd recorded every op from input → loss; this walks it back (chain rule) and writes each weight's contribution into `param.grad`. **Only measures — nothing has moved.**
- **`step()`** — optimizer reads those `.grad`s and updates the weights. *This* is where learning happens.
- Maps to my NumPy MLP: `backward()` ≈ my `backward(...)` returning `dW1,db1…` · `step()` ≈ my `update_params(...)` · `zero_grad()` ≈ nothing (I never accumulated).

### the setup lines
- `torch.randn / ones / zeros` all take a **SHAPE**, not distribution params. `randn(0,1)` is an *empty tensor*, not N(0,1) — `randn` is always standard normal. (Got this wrong twice.)
- Labels shaped `[batch, 1]` — D outputs one number per image. These are **BCE targets**, not MNIST's digit labels.
- `images` = the 32 real digits. `fakes` = the 32 images G just made from noise.

### phase 1 — train D
- **`criterion(prediction, target)`**: prediction = what D *actually said* (`discriminator(images)`); target = what it *should* have said (the labels). Loss = the gap.
- D must see **both** classes. Train it only on fakes and it could output 0 for everything and score perfectly.
- Add the two losses: `backward()` needs one scalar, and grad of a sum = sum of the grads, so both signals reach D.

### phase 2 — train G
- **The line that confused me:** `criterion(discriminator(fakes), real_labels)`. G's output is an *image* — you can't BCE an image against a label. The only thing outputting a number in [0,1] is **D's sigmoid**. So prediction = D's verdict on the fakes; target = the lie, 1.
  - Read it: *"D said this about my fakes; I wish it had said 'real'; the gap is my loss."*
  - **Where's `generator()` in that line?** Nowhere — it already ran. Autograd knows `fakes` came from G, so the gradient flows loss → through D → into `fakes` → into G's weights. **G is judged on work it already did.**
- **Regenerate `fakes` in phase 2.** `d_loss.backward()` frees the graph that made them → reusing gives `RuntimeError: backward through the graph a second time`. (Tutorials often use `.detach()` in phase 1 instead — same problem, other end.)

### why D isn't corrupted in phase 2
`g_loss.backward()` *does* fill gradients on D's weights (the gradient must pass through D to reach G). But `generator_optim.step()` only iterates `generator.parameters()` — **D's `.grad`s are never read**, then wiped by the next `zero_grad()`.
→ **That's why PyTorch needs no "trainable/non-trainable" flag** (unlike the Keras version in my notes). The optimizer's parameter list *is* the flag.

 
### `.detach()` — recurring bug
```python
fakes = generator(noise)  # one forward pass, reused in both phases
 
# phase 1: train D
fake_loss = criterion(discriminator(fakes.detach()), fake_labels)
# detach = D-training shouldn't backprop into G (wasted compute)
 
# phase 2: train G — NOT detached, gradient must flow through D into G's weights
g_loss = criterion(discriminator(fakes), real_labels)
```
- trap Q: `g_loss.backward()` puts gradients on D's weights too (graph passes through D to reach G) — why doesn't D get corrupted?
- A: `generator_optim.step()` only touches G's `.parameters()`. D's gradients just sit unused, cleared next iteration by `discriminator_optim.zero_grad()`
### indentation bug
`if epoch % 10==0:` must sit outside the batch loop, not inside — else prints every batch.
 
### reading loss curves
GAN loss ≠ image quality, just G-vs-D balance. Need real samples, not just numbers, to judge progress.
 
