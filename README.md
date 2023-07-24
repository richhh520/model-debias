# Fast Model DeBias via Machine Unlearning
This repository contains the source code for Fast Model DeBias via Machine Unlearning
## Install
```
pip install -r requirement.txt
```

## Dataset
Construction of Colored MNIST can be referred to https://github.com/clovaai/rebias

Adult can be loaded via Ai fairness 360 https://github.com/Trusted-AI/AIF360

CelebA can be downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and follow the preprocessing in group_DRO (https://github.com/kohpangwei/group_DRO)

removal files are based on https://github.com/facebookresearch/certified-removal

## Experiment on Linear Models
```
python linear_removal.py --data-dir  <your_path>/MNIST --verbose --extractor none --dataset MNIST --train-mode binary --std 10 --lam 1e-3 --num-steps 100
```

## Experiment on C-MNIST
```
python mnist_removal.py --data-dir  <your_path>/MNIST --verbose --extractor none --dataset MNIST --train-mode binary --std 10 --lam 1e-3 --num-steps 100
```

## Experiment on CelebA
```
python celeba_removal.py --verbose --extractor none  --std 10 --lam 1e-3 --num-steps 100
```

## Experiment on LLM

## Acknowledgements
