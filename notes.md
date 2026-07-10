# Notes
some notes jotted down during coding

## divde the components
### components in any ML pipeline (near-universal, especially in PyTorch)
1. Data loader — get data in, clean, ready to feed
2. Model — define the architecture (what transforms input → output)
3. Loss function — define what "wrong" means, numerically
4. Training loop — repeatedly: feed batch → forward pass → compute loss → backward pass → update weights

(optional, add after the above 4 are working)

5. Evaluation — check performance on held-out data / sanity-check outputs (e.g. do generated digits actually look real, not just "loss went down")
6. Inference/deployment — use the trained model on new data, only needed if packaging for others to use

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
-> basically a MLP but output two vectors (mean + standard deviation)

### fun fact (my own opinion)
- Senior engineer = fast pattern-matching across many solved problems (horizontal).
- Senior researcher = knowing exactly where a deep field's unsolved edge sits and pushing past it (vertical).
