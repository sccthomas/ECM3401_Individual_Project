# ECM3401 Individual Project

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

This repository contains the final year project for ECM3401. The project includes work primarily in **Jupyter Notebook**
and **Python**. The goal of the project was to design a novel multi-scale Vision Transformer (ViT) architecture for the
task of semantic segmentation of cancerous cells in histopathology. The project explores the use of self-supervised
learning
(SSL) and Data Augmentation techniques to improve the performance of the ViT model. The project is divided into two main
parts:

1. **Self-Supervised Learning (SSL)**: This part of the project focuses on implementing and evaluating various SSL
   techniques, including contrastive learning and generative SSL methods.
2. **Supervised Learning**: This part of the project focuses on training the ViT model using traditional supervised
   learning techniques.

The two leaning paradigms are compared to evaluate the performance of SSL and Data Augmentation methods in improving the
segmentation
performance, generality, and robustness of the ViT model. Additionally, the project includes a comprehensive set of
experiments and evaluations to further prove any research findings. Causal interventions are performed to understand the
impact of causal feature in input images on the model's predictions. This interventions focus on the True causal
features, the cancerous cells, and non-causal features, the background. The goal of these interventions is to determine
whether the model is still able to make accurate predictions when these features are manipulated.

## Repository Structure

The repository is organized as follows:

```
repo/
├── results/                                            # Model and Training results
│   ├── cell_vit.py                                    
│   ├── contrastive_ssl.py                              
│   ├── contrastive_ssl_finetune.py   
│   ├── generative_ssl.py        
│   ├── generative_ssl_finetune.py              
│   └── standard.py          
├── src/
│   ├── dataset/                      
│       ├── cryonuseg.py                                # CryoNuSeg dataset
│       └── snow.py                                     # Snow dataset
│   ├── self_supervised_learning/     
│       ├── base.py                                     # Base class for self-supervised wrapper models
│       ├── contrastive_loss.py                         # Contrastive SSL Model
│       └── masked_region_loss.py                       # Generative SSL Model
│   ├── training/                     
│       ├── evaluation.py                               # Model evaluation script
│       ├── metrics.py                                  # Training metrics script
│       ├── self_supervised_learning.py                 # Script to train a model with self-supervised learning
│       ├── train.py                                    # Script to train a model with supervised learning
│       └── visualisation.py                            # Script to visualise model internals and other plots
│   └── vision_transformer            
│       ├── common.py                    
│           ├── decoder.py                              # Decoder class for vision transformer model
│           ├── patch_embedding.py                      # Patch embedding class for vision transformer model
│           ├── patch_fusion.py                         # Patch embedding fusion class for vision transformer model
│           ├── swin_transformer_encoder_layer.py       # Swin transformer encoder class for vision transformer models
│           └── transformer_encoder_layer.py            # Transformer encoder class for vision transformer models
│       └── model.py                                    # Vision transformer model
├── tests/                                              # Module tests 
├── GPU_SSL_ViT_Train.ipynb  # Jupyter Notebooks to train a given model with self-supervised learning
├── GPU_ViT_Train.ipynb      # Jupyter Notebooks to train a given model with supervised learning
├── requirements.txt         # Required packages
└── README.md                # Read me file
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sccthomas/ECM3401_Individual_Project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ECM3401_Individual_Project
    ```
3. Set up a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4. Install dependencies (if a `requirements.txt` file exists):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Please refer to the Jupyter Notebooks `GPU_SSL_ViT_Train.ipynb` and `GPU_ViT_Train.ipynb` for instructions on how to
train the models. The notebooks provide step-by-step instructions on how to set up the training process, including
loading the dataset, configuring the model, and training the model with self-supervised or supervised learning.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [sccthomas](https://github.com/sccthomas).