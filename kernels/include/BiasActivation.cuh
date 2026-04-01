#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <unordered_map>
#include <string>

enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    PReLU,
    ELU,
    SELU,
    Softplus,
    Swish,
    Mish,
    HardSigmoid,
    HardTanh,
    HardSwish,
    Softsign,
    Gaussian,
    ArcTan,
    Cube,
    Linear,
    ReLU6,
    LogSigmoid
};



__global__ void forward_bias_activation(const float* d_biases , float* pre_act , float* act , size_t out_features , ActivationType type , float a);