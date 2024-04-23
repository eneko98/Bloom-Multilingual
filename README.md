# Transformer Model Training with BLOOM 7b1

This repository contains scripts and resources for training a transformer model using the BLOOM 7b1 architecture, enhanced with various training techniques such as QLora, Peft, and EarlyStopping. The model is trained on multilingual corpora derived from major dictionaries, targeting improved language model performance.

## Repository Structure

- **Training/**
  - Contains the main training script to train the model on NVIDIA 2x3090 GPUs.
  - **Data/**: Stores training and validation datasets generated by the training script.
  - **Results/**: Includes the saved model checkpoints after training.
- **Corpora/**
  - External corpora used for training can be accessed through the following links:
    - [Oxford Corpus](https://github.com/eneko98/Oxford-Corpus.git)
    - [RAE Spanish Dictionary Corpus](https://github.com/eneko98/RAE-Corpus.git)
    - [EEH Basque Dictionary Corpus (Egungo Euskararen Hiztegia)](https://github.com/eneko98/EEH-Corpus.git)

## Model Training

The model utilizes the `bigscience/bloom-7b1` transformer architecture from Hugging Face. Training is performed on NVIDIA 2x3090 GPUs, completing a total of 10 epochs. Key training enhancements include:

- **QLora**: Quantized Layers for Reduced memory.
- **Peft**: Progressive layer freezing for efficiency.
- **EarlyStopping**: To prevent overfitting and optimize training time.

## Setup and Usage

To set up the training environment and run the training scripts, follow these instructions:

1. **Clone the Repository:**
```
git clone https://github.com/eneko98/Bloom-Multilingual.git
```
```
cd Bloom-Multilingual
```
2. **Install Dependencies:**
```
pip install -r requirements.txt
```
3. **Run Training Script:**
```
python multilingual_training.py
```
4. **Contributing:**
Contributions to this project are welcome. Please fork the repository and submit a pull request to propose changes.