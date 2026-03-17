#pragma once
#include <vector>
#include <functional>
#include "Config.h"



struct Layer {
    std::vector<float> weights;                 // Weights 1d array (out_features * in_features)
    std::vector<float> biases;                  // Biases (out_features)
    std::vector<float> act;                       // Activated outputs of the layer 1d array (batch_size * out_features)
    std::vector<float> pre_act;                       // Pre activated outputs of the layer 1d array (batch_size * out_features)
    std::vector<float> delta;                   // Delta (batch_size * out_features)
    size_t batch_size = 1;                      // Batch size defult 1 (not using batch system)
    std::function<float(float)> activation_function;    // Activation function pointer in the layer
    std::function<float(float)> activation_derivative;  // Activation derivative function pointer in the layer


    Layer(Config&cfg , size_t layer_idx) {
        size_t in_features = (layer_idx == 0) ? cfg.input_size : cfg.neurons_per_layer[layer_idx-1];
        size_t out_features = cfg.neurons_per_layer[layer_idx];
        batch_size = cfg.batch;
        weights.resize(out_features * in_features);
        biases.resize(out_features);
        act.resize(batch_size * out_features);
        pre_act.resize(batch_size * out_features);
        delta.resize(batch_size * out_features);
    };

};