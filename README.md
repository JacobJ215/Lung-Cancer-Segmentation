# Lung-Cancer-Segmentation

![](Screenshots/prediction.png)

## Purpose

The purpose of this project is to enhance lung cancer diagnosis and treatment through automatic tumor segmentation, employing advanced algorithms for precise and efficient detection.

## Background

Lung Cancer, also known as Bronchial Carcinoma, is a prevalent and deadly form of cancer affecting one in 16 individuals worldwide. Accounting for approximately 25% of all cancer-related deaths, it demands improved diagnostic and treatment strategies. Automatic tumor segmentation offers two crucial advantages: reducing the chance of missing tumors during diagnosis and providing essential data on tumor size and volume for staging, assisting medical professionals in devising tailored treatment plans. 

## Dataset
The Medical Segmentation Decathlon dataset is used in this project, consisting of 64 full-body CT scans along with ground truth masks. Each CT scan represents a 3D volume, and the goal is to create a 2D segmentation mask for each slice of the CT scan, identifying tumor regions.

## Preprocessing
The preprocessing step involves loading the CT scan and corresponding label data using the nibabel library. The CT scan data is cropped, starting from slice 30, and then normalized to values between 0 and 1. The preprocessing notebook (preprocessing.ipynb) handles this step, and the preprocessed data is saved in the ../Preprocessed/ directory.

## Dataset and Data Augmentation
The dataset and data augmentation process are implemented in the dataset.py file. The LungDataset class extracts images and corresponding label files from the preprocessed data directory. Data augmentation is performed using the imgaug library, applying affine transformations and elastic transformations to increase the diversity of training data and enhance model generalization.

## Model Architecture
The model architecture used for lung cancer segmentation is based on the U-Net architecture. The U-Net is a popular choice for semantic segmentation tasks, known for its ability to capture both low-level and high-level features effectively. The model is defined in the model.py file and consists of an encoder part with four DoubleConvBlock layers and a decoder part with three DoubleConvBlock layers. The encoder and decoder parts are connected through skip-connections, allowing the model to take advantage of multi-scale feature maps.

![](Screenshots/U-Net.png)

## Training
The training process is implemented in the training.ipynb notebook, utilizing the PyTorch Lightning framework for efficient training. The notebook sets up the data loaders, defines the loss function (binary cross-entropy with logits), and configures the Adam optimizer with a learning rate of 1e-4. The model is trained for 30 epochs with early stopping based on validation loss. To handle class imbalance in the data, a Weighted Random Sampler is used during training.

![](Screenshots/Train_loss.png)

## Evaluation
The model's performance was evaluated using the Dice Score on the validation set, resulting in a low score of 0.0247. This indicates poor segmentation accuracy for lung cancer regions in CT scan images. The low Dice Score implies a high rate of false negatives and false positives, posing risks for misdiagnosis and missed cancer detection. Consequently, further optimization of the model, training process, and validation on independent datasets are necessary before considering clinical applicability.

The obtained result underscores the complexity of lung cancer segmentation and highlights the need for continued research and collaboration with medical experts to improve the model's reliability and effectiveness in accurately identifying tumor regions in medical images.

## Results and Visualization
The trained model's performance is visualized through sample CT scan slices along with their actual masks and predicted masks. The visualizations provide valuable insights into the model's ability to accurately detect tumor regions in medical images.

![](Screenshots/actual_vs_pred.png)

## Conclusion
In conclusion, the lung cancer segmentation project employed deep learning algorithms, including the U-Net architecture and data augmentation techniques, to automatically segment tumor regions in CT scan images. However, the model's performance on the validation set, indicated by the low Dice Score of 0.0247, reveals significant challenges in accurately identifying lung cancer regions. The result highlights the need for further optimization and validation on diverse datasets before considering clinical applicability. As medical image analysis is complex, collaboration with medical experts is crucial to enhance the model's reliability and effectiveness in aiding lung cancer diagnosis and treatment. Caution should be exercised when interpreting the model's predictions, and additional research is essential to advance the model's performance and contribute effectively to the fight against lung cancer.

