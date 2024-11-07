# Image Classification with MLP and CNN: A Comparative Study

## Overview
This project explores the implementation and comparison of **Multilayer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)** for image classification tasks. Using the Fashion MNIST and CIFAR-10 datasets, we analyze the impact of architectural decisions, activation functions, regularization techniques, and optimizer settings on the performance of these models. The study emphasizes the superior performance of CNNs for image classification while providing insights into the behavior of MLPs.

## Features
- **Custom MLP Implementation**: Built from scratch to understand the fundamentals of neural networks.
- **CNN Implementation**: Designed for image classification, leveraging its spatial feature extraction capabilities.
- **Dataset Preprocessing**: Normalization, vectorization, and augmentation for optimal training conditions.
- **Experimental Comparisons**:
  - Weight initialization techniques (e.g., Xavier, Kaiming).
  - Activation functions (ReLU, Sigmoid, Tanh).
  - Regularization methods (L1, L2).
  - Optimizers (SGD with momentum, Adam).
- **Pre-trained ResNet**: Fine-tuned for CIFAR-10 as a benchmark experiment.

## Datasets
1. **Fashion MNIST**:
   - Grayscale images of 10 fashion categories (e.g., T-shirts, bags, sneakers).
   - Preprocessed for MLP and CNN compatibility.
2. **CIFAR-10**:
   - Colored images spanning 10 classes (e.g., airplanes, cats, trucks).
   - More complex dataset for rigorous model testing.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mlp-cnn-comparison.git
   cd mlp-cnn-comparison
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running Models
1. **Train and evaluate MLP**:
   ```bash
   python mlp_train.py --dataset fashion_mnist
   ```
2. **Train and evaluate CNN**:
   ```bash
   python cnn_train.py --dataset cifar10
   ```
3. **Fine-tune ResNet**:
   ```bash
   python resnet_finetune.py --dataset cifar10
   ```

### Notebook
Interactive exploration of results:
```bash
jupyter notebook analysis.ipynb
```

## Results
- **Fashion MNIST**:
  - **MLP Accuracy**: ~87.31%
  - **CNN Accuracy**: ~91.69%
- **CIFAR-10**:
  - **MLP Accuracy**: ~39.41%
  - **CNN Accuracy**: ~70.89%
- **Optimizers on CIFAR-10**:
  - Adam: Best accuracy (71.56%) and stability.
  - SGD with momentum (0.9): Fastest convergence.

## Contributors
This project was collaboratively developed by:
- **Raphael Fontaine** (261051635)
- **Mohammad Nafar** (260899098)
- **Winnie Yongru Pan** (261001758)

## Contributing
We welcome contributions! To propose changes:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your fork and open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Fashion MNIST and CIFAR-10 Datasets**: Provided by [PyTorch](https://pytorch.org/vision/).
- **ResNet Model**: For transfer learning insights.
- **COMP 551**: Guidance and support throughout the project.
