# 🧠 Brain Tumor Classification Model

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3126/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project implements and compares several deep learning models for the detection and classification of brain tumors using MRI scan images. The models are trained on the Brain Tumor MRI Dataset from Kaggle, enabling accurate identification of different types of brain tumors to assist medical professionals in diagnosis. This repository contains the code for the models, training scripts, and helper functions.

## 🗃️ Dataset

The project uses the Brain Tumor MRI Dataset from Kaggle, which includes:
- A comprehensive collection of brain MRI scans
- Four tumor categories for classification: Glioma, Meningioma, Pituitary, and No Tumor.

Dataset Source: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Dataset Statistics
- **Training Set:** 5712 images
- **Testing Set:** 1311 images
- **Total Images:** 7023 MRI scans
- **Classes:** 
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor (Normal)

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/BrainTumorModels.git
    cd BrainTumorModels
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install required dependencies:**
    ```bash
    pip install torch torchvision matplotlib
    ```

## 🚀 How to Run

1.  **Download the dataset:** Download and extract the dataset from the Kaggle link above into a directory of your choice.

2.  **Update dataset paths:** Open `importing_dataset.py` and update the `directory_train_dataset` and `directory_test_dataset` variables to point to your training and testing data folders.

3.  **Train a model:** Run one of the model files to train it. For example, to train `model_2`:
    ```bash
    python model_2.py
    ```

4.  **Test a model:** Run one of the testing scripts to evaluate a trained model. For example, to test `model_2`:
    ```bash
    python testing_model_2.py
    ```

## 📂 Project Structure

```
├── .gitignore
├── README.md
├── helper_functions.py
├── importing_dataset.py
├── model_0.py
├── model_1.py
├── model_2.py
├── model_3.py
├── test.py
├── testing_model_2.py
├── testing_model_3.py
└── trained_models_data/
    ├── model_0_checkpoint.pth
    ├── model_1_checkpoint.pth
    ├── model_2_checkpoint.pth
    └── model_3_checkpoint.pth
```

-   `helper_functions.py`: Contains helper functions for plotting, training, and evaluation.
-   `importing_dataset.py`: Handles loading and transforming the dataset.
-   `model_*.py`: Defines the different model architectures (Model 0, 1, 2, 3).
-   `testing_model_*.py`: Scripts to test the trained models.
-   `trained_models_data/`: Directory where trained model checkpoints are saved.

## 📊 Model Architectures

This project explores multiple CNN architectures.

### Model 2 Architecture

This model consists of two convolutional blocks followed by a linear layer stack.

```python
class BrainTumorModelV2(nn.Module):
    def __init__(self,input_shape_flattened,input_channel,output_shape,hidden_units):
        super().__init__()
        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.Linear_layer_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape_flattened, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    def forward(self,x):
        x=self.conv_block_1(x)
        x=self.conv_block_2(x)
        x=self.Linear_layer_stack(x)
        x=nn.Softmax(dim=1)(x)
        return x
```

### Model 3 Architecture

Our best performing model (Model 3) uses a deeper CNN architecture.

-   **Accuracy**: 92-93% on test set
-   **Training Time**: ~2.5 hours on GPU

## 📈 Performance

The models' performance is evaluated based on accuracy. Model 3 is the top-performing model.

| Model   | Test Accuracy | Details                               |
| ------- | ------------- | ------------------------------------- |
| Model 0 | ~75%          | Baseline model with a simple architecture. |
| Model 1 | ~85%          | Increased complexity from Model 0.    |
| Model 2 | ~90%          | Deeper architecture, better performance. |
| Model 3 | **~93%**      | Most complex and best performing model. |


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

<div align="center">
Made with ❤️ for advancing medical imaging analysis
</div>
