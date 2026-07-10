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
### resources:
- [MNIST documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)

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
```python
data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform.ToTensor())
```
- `root='./data'` — creates a folder called `data`
- `train=True` — True/False corresponds to 60k / 10k images. Use more data (60k) to train, less (10k) to test.
- `transform=transforms.ToTensor()` — data loads as PIL (python imaging library), but PyTorch only works with tensors, so transform to tensor
  - reshapes PIL's (H, W, C) → PyTorch's (C, H, W)
  - rescales 0-255 ints → 0.0-1.0 floats → no need to normalize anymore
- `download=True` — puts it in root directory (data folder)

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
plt.title(f'Label: {label}')
plt.show()
```
