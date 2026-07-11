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

## 1. data loader

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

## 2. model
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
