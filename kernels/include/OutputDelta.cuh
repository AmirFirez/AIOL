#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <unordered_map>
#include <string>
#include "Activations.cuh"

__global__ void output_delta(float* act,float* pre_act,ActivationType type,int out_features,float* delta, float * labels);
