#include <cuda_runtime.h>
#include <cmath>
#include "BiasActivation.cuh"
#include "Activations.cuh"




__global__ void forward_bias_activation(const float* d_biases , float* pre_act ,float* act , const size_t out_features,ActivationType type , float a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // Index for which weight we should be at 
    int batch = blockIdx.y;                             // Index for which batch we should be in

    if (idx >= out_features) return;

    size_t neuron = batch * out_features + idx;
    pre_act[neuron] += d_biases[idx];
    float neuron_value = pre_act[neuron];

    act[neuron] = activate(type,neuron_value,a);

}

