#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <unordered_map>
#include <string>
#include "Activations.cuh"




__global__ void forward_bias_activation(const float* d_biases , float* pre_act , float* act , size_t out_features , ActivationType type , float a);