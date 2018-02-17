# revnet

[PyTorch](http://pytorch.org/) implementation of [the reversible residual
network](https://arxiv.org/abs/1707.04585).


## Requirements

The main requirement ist obviously [PyTorch](http://pytorch.org/). CUDA is
strongly recommended.

The training script requires [tqdm](https://pypi.python.org/pypi/tqdm) for the
progress bar.


## Note

The revnet models in this project tend to have exploding gradients. To
counteract this, I used gradient norm clipping. For the experiments below you
would call the following command:

```
python train_cifar.py --model revnet38 --clip 0.25
```


## Results

### CIFAR-10

| Model    | Accuracy | Memory Usage | Params |
|----------|----------|--------------|--------|
| revnet38 | 91.98%   | 660 MB       | 0.47 M |
