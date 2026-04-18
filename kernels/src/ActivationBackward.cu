#include <cuda_runtime.h>
#include <cmath>
#include "Activations.cuh"


__global__ void activation_backward(float* d_grad_buffer , const size_t out_features , const ActivationType type,float* x,float a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;

    if (idx >= out_features) return;

    size_t neuron = batch * out_features + idx;

    d_grad_buffer[neuron] = activate_derivative(type,x[neuron],a);


}