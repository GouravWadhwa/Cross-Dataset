# Cross-Dataset

## Summary

This repository contains the tensorflow implementation of Cross-Domain Adaptation for Classification model.
Here we use the PACS dataset which can be downloaded from [here] (https://drive.google.com/drive/folders/1SKvzI8bCqW9bcoNLNCrTGbg7gBSw97qO)

The PACS dataset contains images from 4 domains, namely Image, Art painting, Sketch and Carton with 7 classes. We treat the Sketch domain as target dataset and other three domains as source dataset.

For Training we use a combination of three losses
* Cross Entropy Loss for source dataset
* Cross Entropy Loss for target dataset
* Entropy Loss

## Installation

```bash
git clone https://github.com/GouravWadhwa/Cross-Dataset.git
cd Hypergraphs-Image-Inpainting
```

## Training and Testing

For generating the training (train_file.txt), the validation (val_file.txt), and the  testing (test_file.txt) sets run the following code:

```bash
python create_train_data_file.py
```

For training a new model run the following code:

```bash
python train.py
```

For finding the testing accuracy run the following code:

```bash
python test.py
```

## Results

Following are the hyperparameters for training
* λ_1 = 0.9
* λ_2 = 1.0
* λ_3 = 0.1

Validation Accuracy - 91.92%
Testing Accuracy - 85%

