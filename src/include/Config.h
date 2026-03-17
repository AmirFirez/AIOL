#pragma once
#include <vector>
#include <string>

struct Config {
    size_t input_size;                          // Number of input features
    size_t layer_count;                         // Number of layers
    size_t batch;                               // Batch size
    std::vector<size_t> neurons_per_layer;      // Number of neurons per layer
    std::string loss;                           // Loss function name ("MSE","BCE",etc)
    std::vector<std::string> activation_func_per_layer; // Activation function name per layer
    float alpha;                                // Optional hyperparameter for ("LeakyReLU","Huber",etc)
    float learning_rate;                        // Learning rate
};