# miniMNIST-cpp

This project implements a **minimal** neural network in C++ (clang++) for classifying handwritten digits from the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download). The entire implementation is  **~270 lines of code**.

## Features

- Two-layer neural network (input → hidden → output)
- ReLU activation function for the hidden layer
- Softmax activation function for the output layer
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD) optimizer

## Performance

```bash
Epoch 1, Accuracy: 94.4583%, Avg Loss: 0.299816, Time: 2.484 seconds
Epoch 2, Accuracy: 95.85%, Avg Loss: 0.135207, Time: 2.54 seconds
Epoch 3, Accuracy: 96.525%, Avg Loss: 0.0925771, Time: 2.543 seconds
Epoch 4, Accuracy: 96.8083%, Avg Loss: 0.0687637, Time: 2.545 seconds
Epoch 5, Accuracy: 96.9917%, Avg Loss: 0.0524336, Time: 2.55 seconds
Epoch 6, Accuracy: 97.05%, Avg Loss: 0.0404643, Time: 2.555 seconds
Epoch 7, Accuracy: 97.2167%, Avg Loss: 0.0313959, Time: 2.555 seconds
Epoch 8, Accuracy: 97.3583%, Avg Loss: 0.0246683, Time: 2.551 seconds
Epoch 9, Accuracy: 97.4667%, Avg Loss: 0.0193795, Time: 2.56 seconds
Epoch 10, Accuracy: 97.5583%, Avg Loss: 0.0154899, Time: 2.564 seconds
Epoch 11, Accuracy: 97.5583%, Avg Loss: 0.0125247, Time: 2.56 seconds
Epoch 12, Accuracy: 97.5917%, Avg Loss: 0.0101825, Time: 2.558 seconds
Epoch 13, Accuracy: 97.6417%, Avg Loss: 0.00837908, Time: 2.582 seconds
Epoch 14, Accuracy: 97.6667%, Avg Loss: 0.00693848, Time: 2.567 seconds
Epoch 15, Accuracy: 97.6917%, Avg Loss: 0.00580494, Time: 2.56 seconds
Epoch 16, Accuracy: 97.6833%, Avg Loss: 0.00494942, Time: 2.553 seconds
Epoch 17, Accuracy: 97.65%, Avg Loss: 0.00428182, Time: 2.568 seconds
Epoch 18, Accuracy: 97.6667%, Avg Loss: 0.00373187, Time: 2.578 seconds
Epoch 19, Accuracy: 97.65%, Avg Loss: 0.00329977, Time: 2.56 seconds
Epoch 20, Accuracy: 97.6583%, Avg Loss: 0.00294499, Time: 2.558 seconds
```

## Prerequisites

- clang++ compiler
- MNIST dataset files:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`

## Compilation

```bash
clang++ -O3 -march=native -ffast-math -std=c++17 -o nn nn.cpp -lm
```

## Usage

1. Place the MNIST dataset files in the `data/` directory.
2. Compile the program.
3. Run the executable:

   ```bash
   ./nn
   ```

The program will train the neural network on the MNIST dataset and output the accuracy and average loss for each epoch.

## Configuration

You can adjust the following parameters in `nn.cpp`:

- `HIDDEN_SIZE`: Number of neurons in the hidden layer
- `LEARNING_RATE`: Learning rate for SGD
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Mini-batch size for training
- `TRAIN_SPLIT`: Proportion of data used for training (the rest is used for testing)

## License

This project is open-source and available under the [MIT License](LICENSE).
