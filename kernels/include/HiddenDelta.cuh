#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "Activations.cuh"

__global__ void hidden_delta(float* weights,ActivationType type,float* pre_act,int out_features,float* delta,float* weights_next ,int in_features_next ,int out_features_next, float* delta_next);

