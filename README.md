# Fast Model DeBias via Machine Unlearning
This repository contains the source code for Fast Model DeBias with Machine Unlearning in NIPS2023

## Abstract
Recent discoveries have revealed that deep neural networks might behave in a biased manner in many real-world scenarios. For instance, deep networks trained on a large-scale face recognition dataset CelebA tend to predict blonde hair for females and black hair for males. Such biases not only jeopardize the robustness of models but also perpetuate and amplify social biases, which is especially concerning for automated decision-making processes in healthcare, recruitment, etc., as they could exacerbate unfair economic and social inequalities among different groups. Existing debiasing methods suffer from high costs in bias labeling or model re-training, while also exhibiting a deficiency in terms of elucidating the origins of biases within the model. To this respect, we propose a fast model debiasing framework (FMD) which offers an efficient approach to identify, evaluate and remove biases inherent in trained models. The FMD identifies biased attributes through an explicit counterfactual concept and quantifies the influence of data samples with influence functions. Moreover, we design a machine unlearning-based strategy to efficiently and effectively remove the bias in a trained model with a small counterfactual dataset.

## Environment Install
```
pip install -r requirement.txt
```

## Dataset Preparation
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

## Experiment on Adult
```
python adult_removal.py --verbose --extractor none  --std 10 --lam 1e-3 --num-steps 100
```
