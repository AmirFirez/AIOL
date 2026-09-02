#include <cuda_runtime.h>
#include <cmath>
#include "Activations.cuh"

__global__ void output_delta(float* act,float* pre_act,ActivationType type,int out_features,float* delta, float * labels){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;

    if (idx >= out_features) return;

    size_t neuron = batch * out_features + idx;
    delta[neuron] = (act[neuron] - labels[neuron]) * activate_derivative(type,pre_act[neuron]);

}