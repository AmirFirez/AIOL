#include <cuda_runtime.h>
#include <cmath>
#include "Activations.cuh"
#include "HiddenDelta.cuh"

__global__ void hidden_delta(float* weights,ActivationType type,float* pre_act,int out_features,float* delta,float* weights_next ,int in_features_next ,int out_features_next, float* delta_next) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;

    if (idx >= out_features) return;

    size_t neuron = batch * out_features + idx;

    float sum = 0;

    for (size_t i = 0;i < out_features_next;i++) {
        sum += weights_next[i * in_features_next + idx] * delta_next[batch * out_features_next + i]; 
    }

    delta[neuron] = activate_derivative(type,pre_act[neuron]) * sum;

}