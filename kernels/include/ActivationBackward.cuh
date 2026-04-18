#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "Activations.cuh"


__global__ void activation_backward(float* d_grad_buffer , const size_t out_features , const ActivationType type,float* x,float a);