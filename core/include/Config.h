#pragma once
#include <vector>
#include <string>
#include <BiasActivation.cuh>
#include <WeightInit.h>

struct Config {
    size_t input_size;                          // Number of input features
    size_t layer_count;                         // Number of layers
    int batch_size;                                  // Batch size
    std::vector<size_t> neurons_per_layer;      // Number of neurons per layer
    std::string loss;                           // Loss function name ("MSE","BCE",etc)
    WeightsType WeightsType;                   // Weights random generator type ("HE" , "XavierNormal" , etc)
    std::vector<ActivationType> activation_func_per_layer; // Activation function name per layer
    float alpha;                                // Optional hyperparameter for ("LeakyReLU","Huber",etc)
    float learning_rate;                        // Learning rate
    float beta;                                 // Beta
    float a;                                    // Used in some functions like ("PReLU" , "ELU" , etc)
};