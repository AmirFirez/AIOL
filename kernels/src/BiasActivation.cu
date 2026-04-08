#include <cuda_runtime.h>
#include <cmath>
#include "BiasActivation.cuh"




__global__ void forward_bias_activation(const float* d_biases , float* pre_act ,float* act , const size_t out_features,ActivationType type , float a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // Index for which weight we should be at 
    int batch = blockIdx.y;                             // Index for which batch we should be in

    if (idx >= out_features) return;

    size_t neuron = batch * out_features + idx;
    pre_act[neuron] += d_biases[idx];
    float neuron_value = pre_act[neuron];

    switch (type) {
        case Sigmoid:
            act[neuron] = 1 / (1 + expf(-neuron_value));
            break;

        case Tanh:
            act[neuron] = tanhf(neuron_value);
            break;

        case ReLU:
            act[neuron] = fmaxf(0,neuron_value);
            break;

        case LeakyReLU:
            act[neuron] = fmaxf(a * neuron_value,neuron_value);
            break;
            
        case PReLU:
            act[neuron] = fmaxf(a * neuron_value,neuron_value);
            break;

        case ELU:
            act[neuron] = (neuron_value >= 0) ? neuron_value : a * (expf(neuron_value) - 1);
            break;

        case SELU:
            act[neuron] = 1.0507 * ((neuron_value >= 0) ? neuron_value : 1.67326 * (expf(neuron_value) - 1));
            break;
        
        case Softplus:
            act[neuron] = logf(1 + expf(neuron_value));
            break;

        case Swish:
            act[neuron] = neuron_value / (1 + expf(-neuron_value));
            break;

        case Mish:
            act[neuron] = neuron_value * tanhf(logf(1 + expf(neuron_value)));
            break;

        case HardSigmoid:
            act[neuron] = fmaxf(0,fminf(1,0.2 * neuron_value + 0.5));
            break;

        case HardTanh:
            act[neuron] = fmaxf(-1,fminf(1,neuron_value));
            break;

        case HardSwish:
            act[neuron] = neuron_value * fmaxf(0.0f, fminf(1.0f, (neuron_value + 3.0f) / 6.0f));
            break;

        case Softsign:
            act[neuron] = neuron_value / (1 + fabsf(neuron_value));
            break;
        
        case Gaussian:
            act[neuron] = expf(-(neuron_value*neuron_value));
            break;

        case ArcTan:
            act[neuron] = atanf(neuron_value);
            break;

        case Cube:
            act[neuron] = neuron_value * neuron_value * neuron_value;
            break;

        case Linear:
            act[neuron] = neuron_value;
            break;
        
        case ReLU6:
            act[neuron] = fminf(fmaxf(0,neuron_value),6);
            break;
        
        case LogSigmoid:
            act[neuron] = -logf(1 + expf(-neuron_value));
            break;

        default:
            act[neuron] = neuron_value;
            break;

    }


}

