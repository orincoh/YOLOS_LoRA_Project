# Enhancing Object Detection in YOLOS-Small through Advanced LoRA Methods

## Overview

This project examines the integration of Low-Rank Adaptation (LoRA) methods into YOLOS-Small, a Vision Transformer-based object detection model. The goal is to explore whether fine-tuning techniques can improve computational efficiency while maintaining detection accuracy on the COCO 2017 dataset.

The YOLOS-Small model, sourced from [Hugging Face's YOLOS-Small repository](https://huggingface.co/hustvl/yolos-small), was used as the baseline. The study evaluates various LoRA-based approaches, including LoRA, AdaLoRA, LoHa, and LoKr, to determine their impact on model performance.

This project was developed as part of the Deep Learning course at Ben-Gurion University (2025).

## Dataset

- The dataset used in this project is COCO 2017, accessed through Hugging Face Datasets (`detection-datasets/coco`).
- The dataset underwent preprocessing using the YOLOS image processor to ensure compatibility with the model.
- A subset of 2,000 training images and 900 test images was selected for experimentation.

## LoRA Variants Implemented

| LoRA Variant | Rank (r) | Scaling (α) | Dropout | Special Features        |
| ------------ | -------- | ----------- | ------- | ----------------------- |
| LoRA         | 16       | 8           | 0.5     | Standard LoRA           |
| AdaLoRA      | 16 → 12  | Adaptive    | None    | Dynamic rank allocation |
| LoHa         | 12       | 8           | 0.3     | Rank + module dropout   |
| LoKr         | 16       | 8           | 0.4     | Kronecker decomposition |

## Evaluation & Results

The results indicate that there was no significant improvement over the baseline model, though a slight enhancement was observed. This suggests that with further refinements, such as better hyperparameter tuning and a larger dataset, LoRA-based methods have the potential to improve performance further.

## Project Files

This repository contains the following files and directories:

- **`Enhancing_YOLOS_S_through_Advanced_LoRA_Methods.py`** – The main script containing the implementation of the LoRA-based YOLOS-Small fine-tuning. This script is fully runnable and includes all necessary steps from data preprocessing to model training and evaluation.
- **`Models with LoRA Variants/`** – Contains the trained YOLOS-Small models after fine-tuning:
  - `yolos_lora.pth`
  - `yolos_adalora.pth`
  - `yolos_loha.pth`
  - `yolos_lokr.pth`
  - These models can be loaded directly in the corresponding sections of the main script.
- **`Visualizations/`** – Graphs and visualizations generated during training and evaluation.
- **`Enhancing YOLOS-S through Advanced LoRA Methods.pdf`** – The final project report summarizing the findings.
- **`Part1_setup_and_data_understanding.ipynb` → `Part7_models_comparison.ipynb`**  
  - These **seven Jupyter Notebook files** are structured **for display purposes only** and include outputs of different steps in the process.  
  - They **are not independently runnable**, as each part depends on the previous ones.  
  - The full executable code is available in **`Enhancing_YOLOS_S_through_Advanced_LoRA_Methods.py`**.

## Future Work

- Improve hyperparameter optimization to better leverage LoRA-based adaptations.
- Explore additional fine-tuning strategies to enhance object detection performance.
- Increase the dataset size to allow for more comprehensive training and evaluation.

## Authors

- Orin Cohen ([orincoh@post.bgu.ac.il](mailto:orincoh@post.bgu.ac.il))
- Tom Damari ([damarit@post.bgu.ac.il](mailto:damarit@post.bgu.ac.il))

