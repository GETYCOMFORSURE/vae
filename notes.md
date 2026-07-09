# Notes
some notes jotted down during coding

## building data loader
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

### sections within data loader: all supervised/unsupervised pipeline needs these (in order)
- Get raw data into memory (load)
- Separate data you train on from data you evaluate on (split) — do this before normalizing, so you don't leak test-set information into your normalization stats
- Make sure values are in a range the model can learn from efficiently (normalize) — because gradient descent behaves badly on wildly different feature scales; compute mean/std from the training set only
- Randomize order (shuffle) — because if data comes in a meaningful order (e.g. sorted by label), the model can learn spurious patterns from the order itself instead of the actual features
- Feed data to the model in small groups, not all at once (batch) — memory can't hold the whole dataset at once, and the model updates its weights after each small batch (not once per epoch), giving many fast, stable updates instead of one huge slow one
