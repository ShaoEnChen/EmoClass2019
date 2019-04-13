# EmoClass2019
Emotion Classification 2019 trying different state-of-the-art machine learning techniques

### Data Description
FER2013 is a large-scale dataset collected automatically by the Google image search API. It contains 28,709 training images and 3,589 test images with seven expression labels including anger, disgust, fear, happiness, sadness, surprise and neutral. Each data consists of a grayscale 48 * 48 pixels face image converted into a string of pixels. The dataset is originally prepared by Pierre-Luc Carrier and Aaron Courville, as part of their research project, and introduced during the ICML 2013 Challenges in Representation Learning.

### Problem Outcomes
We plan to explore and generate different features and demonstrate the outcome’s dependency on these features and we hope to achieve state-of-the-art emotion classification accuracy on facial images. Experiments will be conducted to compare our approach with the baseline approaches. 

### Dependencies
- Python 3.5
- PyTorch 1.0

### Train and evaluate model
    python main.py --model VGG19 --bs 64 --lr 0.01 --save-path checkpoints/best_model_vgg19.t7

### Results
- Model:    VGG19; the best validation accuracy: 70.967%; testing accuracy: 72.834%
- Model: Resnet18; the best validation accuracy: 71.441%; testing accuracy: 72.778% 
