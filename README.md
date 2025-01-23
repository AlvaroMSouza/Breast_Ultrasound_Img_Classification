# Breast_Ultrasound_Img_Classification

## About the Dataset
Breast cancer is one of the most common causes of death among women worldwide. Early detection is crucial for reducing mortality rates. This project uses the Breast Ultrasound Images Dataset, which contains ultrasound images categorized into three classes: normal, benign, and malignant. The dataset includes 780 images with an average size of 500x500 pixels in PNG format. Each image is accompanied by a ground truth mask.

### Dataset Details
- *Source*: Collected in 2018 from 600 female patients aged between 25 and 75 years.
- *Classes*: Normal, Benign, Malignant.
- *Image Format*: PNG.
- *Ground Truth*: Provided as masks for segmentation tasks.

 
## Project Overview

This project aims to classify breast ultrasound images into three categories: *normal*, *benign*, and *malignant*. Two deep learning models were implemented and evaluated:

1. Custom CNN: A convolutional neural network designed from scratch.
2. ResNet-50: A pre-trained ResNet-50 model fine-tuned for this task.

The models were trained and tested on the Breast Ultrasound Images Dataset, and their performance was evaluated using metrics such as accuracy, precision, and recall.

## Code Structure
The project is implemented in Python using PyTorch. Below is an overview of the code:

### Key Libraries Used
- *PyTorch*: For building and training deep learning models.
- *Torchvision*: For data loading, transformations, and pre-trained models.
- *Matplotlib*: For visualizing images and results.
- *Torchmetrics*: For computing evaluation metrics (accuracy, precision, recall).

### Code Workflow
1. *Data Loading and Preprocessing*:
- Images are resized to 224x224 pixels.
- Data augmentation (random horizontal flip, rotation, and autocontrast) is applied to the training set.
- Images are normalized using ImageNet mean and standard deviation.
- Ground truth masks are excluded from the dataset.

*Dataset Splitting*:
- The dataset is split into training (70%) and testing (30%) sets.

*Model Architecture*:
- Custom CNN: A sequential CNN with convolutional, ELU activation, max-pooling, and fully connected layers.
- ResNet-50: A pre-trained ResNet-50 model with the final fully connected layer replaced for 3-class classification.

*Training*:
- Both models are trained using the Adam optimizer and CrossEntropyLoss.
- Training is performed for 25 epochs.

*Evaluation*:
- Models are evaluated on the test set using accuracy, precision, and recall.

*Model Saving*:
- Trained models are saved as .pth files for future use.

## Usage Instructions
### Prerequisites
- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Torchmetrics

### Installation
1. Clone the repository:
git clone https://github.com/your-username/breast-cancer-classification.git
cd breast-cancer-classification

2. Install the required libraries:
pip install torch torchvision matplotlib torchmetrics

3. Download the dataset from Kaggle and place it in the Dataset_BUSI_with_GT folder.

## Running the Code
1. Train and evaluate the models:
python train.py

2. Visualize misclassified images (optional):
- Modify the code to include visualization functions as described in the project.

## Results

### Performance Metrics
*Custom CNN*:
- Accuracy: 55%
- Precision: 55%
- Recall: 55%

*ResNet-50*:
- Accuracy: 65%
- Precision: 65%
- Recall: 65%

### Visualizations
- *Training Loss*: Plot of training loss over epochs for both models.
- *Misclassified Images*: Examples of images that were incorrectly classified by the models.

## Recomendations
Try building a new architecture for the CNN Model in order to see if the results improved, besides that try to increase the epoch size in training.

## Credits
- *Dataset*: Breast Ultrasound Images Dataset by Al-Dhabyani et al.
- *Libraries*: PyTorch, Torchvision, Matplotlib, Torchmetrics.



