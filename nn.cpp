#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0005f
#define MOMENTUM 0.9f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

constexpr size_t HIDDEN_WEIGHTS_SIZE = INPUT_SIZE * HIDDEN_SIZE;
constexpr size_t OUTPUT_WEIGHTS_SIZE = HIDDEN_SIZE * OUTPUT_SIZE;

struct alignas(32) Layer {
    float* weights;
    float* biases;
    float* weight_momentum;
    float* bias_momentum;
    const int input_size;
    const int output_size;
    const size_t weights_size;

    Layer(int in_size, int out_size)
        : input_size(in_size), output_size(out_size),
          weights_size(in_size * out_size) {

        weights = new float[weights_size];
        biases = new float[out_size]();
        weight_momentum = new float[weights_size]();
        bias_momentum = new float[out_size]();

        const float scale = std::sqrt(2.0f / in_size);
        for (size_t i = 0; i < weights_size; i++) {
            weights[i] = ((float)std::rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }

    void forward(const float* __restrict__ input, float* __restrict__ output) const {
        std::copy(biases, biases + output_size, output);

        for (int j = 0; j < input_size; j++) {
            const float input_val = input[j];
            const float* weights_row = weights + j * output_size;
            for (int i = 0; i < output_size; i++) {
                output[i] += input_val * weights_row[i];
            }
        }

        // Vectorizable ReLU
        for (int i = 0; i < output_size; i++) {
            output[i] = std::max(0.0f, output[i]);
        }
    }

    void backward(const float* __restrict__ input,
                 const float* __restrict__ output_grad,
                 float* __restrict__ input_grad,
                 const float lr) {

        if (input_grad) {
            std::fill(input_grad, input_grad + input_size, 0.0f);
            for (int j = 0; j < input_size; j++) {
                float sum = 0.0f;
                const float* weights_row = weights + j * output_size;
                for (int i = 0; i < output_size; i++) {
                    sum += output_grad[i] * weights_row[i];
                }
                input_grad[j] = sum;
            }
        }

        // Update weights and biases
        for (int j = 0; j < input_size; j++) {
            const float input_val = input[j];
            float* weights_row = weights + j * output_size;
            float* momentum_row = weight_momentum + j * output_size;

            for (int i = 0; i < output_size; i++) {
                const float grad = output_grad[i] * input_val;
                momentum_row[i] = MOMENTUM * momentum_row[i] + lr * grad;
                weights_row[i] -= momentum_row[i];
            }
        }

        for (int i = 0; i < output_size; i++) {
            bias_momentum[i] = MOMENTUM * bias_momentum[i] + lr * output_grad[i];
            biases[i] -= bias_momentum[i];
        }
    }

    ~Layer() {
        delete[] weights;
        delete[] biases;
        delete[] weight_momentum;
        delete[] bias_momentum;
    }
};

class Network {
    Layer hidden;
    Layer output;
    float hidden_output[HIDDEN_SIZE];
    float final_output[OUTPUT_SIZE];
    float output_grad[OUTPUT_SIZE];
    float hidden_grad[HIDDEN_SIZE];

public:
    Network() : hidden(INPUT_SIZE, HIDDEN_SIZE), output(HIDDEN_SIZE, OUTPUT_SIZE) {}

    const float* train(const float* input, const int label, const float lr) {
        hidden.forward(input, hidden_output);
        output.forward(hidden_output, final_output);
        softmax(final_output);

        // Compute gradients
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_grad[i] = final_output[i] - (i == label);
        }

        output.backward(hidden_output, output_grad, hidden_grad, lr);

        // ReLU derivative
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hidden_grad[i] *= (hidden_output[i] > 0);
        }

        hidden.backward(input, hidden_grad, nullptr, lr);
        return final_output;
    }

    int predict(const float* input) {
        hidden.forward(input, hidden_output);
        output.forward(hidden_output, final_output);
        softmax(final_output);
        return std::max_element(final_output, final_output + OUTPUT_SIZE) - final_output;
    }

private:
    static void softmax(float* input) {
        const float max_val = *std::max_element(input, input + OUTPUT_SIZE);
        float sum = 0.0f;

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            input[i] = std::exp(input[i] - max_val);
            sum += input[i];
        }

        const float inv_sum = 1.0f / sum;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            input[i] *= inv_sum;
        }
    }
};

class MNISTDataset {
    std::vector<float> images;
    std::vector<unsigned char> labels;
    const size_t n_images;
    std::mt19937 rng{std::random_device{}()};

public:
    MNISTDataset(const std::string& img_path, const std::string& lbl_path)
        : n_images(read_data(img_path, lbl_path)) {}

    void shuffle() {
        std::vector<size_t> indices(n_images);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<float> temp_images = images;
        std::vector<unsigned char> temp_labels = labels;

        for (size_t i = 0; i < n_images; i++) {
            const size_t idx = indices[i];
            std::copy(temp_images.begin() + idx * INPUT_SIZE,
                     temp_images.begin() + (idx + 1) * INPUT_SIZE,
                     images.begin() + i * INPUT_SIZE);
            labels[i] = temp_labels[idx];
        }
    }

    size_t size() const { return n_images; }
    const float* get_image(size_t idx) const { return images.data() + idx * INPUT_SIZE; }
    unsigned char get_label(size_t idx) const { return labels[idx]; }

private:
    size_t read_data(const std::string& img_path, const std::string& lbl_path) {
        std::ifstream img_file(img_path, std::ios::binary);
        std::ifstream lbl_file(lbl_path, std::ios::binary);

        if (!img_file || !lbl_file) {
            std::cerr << "Failed to open MNIST files\n";
            exit(1);
        }

        int magic_number, n_images, n_rows, n_cols;
        img_file.read(reinterpret_cast<char*>(&magic_number), 4);
        img_file.read(reinterpret_cast<char*>(&n_images), 4);
        img_file.read(reinterpret_cast<char*>(&n_rows), 4);
        img_file.read(reinterpret_cast<char*>(&n_cols), 4);

        n_images = __builtin_bswap32(n_images);

        std::vector<unsigned char> temp_images(n_images * INPUT_SIZE);
        images.resize(n_images * INPUT_SIZE);
        labels.resize(n_images);

        img_file.read(reinterpret_cast<char*>(temp_images.data()), n_images * INPUT_SIZE);

        lbl_file.seekg(8);
        lbl_file.read(reinterpret_cast<char*>(labels.data()), n_images);

        // Convert to float once during loading
        for (size_t i = 0; i < n_images * INPUT_SIZE; i++) {
            images[i] = temp_images[i] / 255.0f;
        }

        return n_images;
    }
};

int main() {
    Network net;
    MNISTDataset dataset(TRAIN_IMG_PATH, TRAIN_LBL_PATH);
    dataset.shuffle();

    const size_t train_size = static_cast<size_t>(dataset.size() * TRAIN_SPLIT);
    const size_t test_size = dataset.size() - train_size;
    constexpr float learning_rate = LEARNING_RATE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        const auto start = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;

        for (size_t i = 0; i < train_size; i++) {
            const float* img_data = dataset.get_image(i);
            const auto final_output = net.train(img_data, dataset.get_label(i), learning_rate);
            total_loss += -std::log(final_output[dataset.get_label(i)] + 1e-10f);
        }

        size_t correct = 0;
        for (size_t i = train_size; i < dataset.size(); i++) {
            const float* img_data = dataset.get_image(i);
            correct += (net.predict(img_data) == dataset.get_label(i));
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

        std::cout << "Epoch " << (epoch + 1)
                  << ", Accuracy: " << (static_cast<float>(correct) / test_size * 100)
                  << "%, Avg Loss: " << (total_loss / train_size)
                  << ", Time: " << duration << " seconds\n";
    }

    return 0;
}