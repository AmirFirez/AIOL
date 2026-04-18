#pragma once
#include <cmath>

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

__device__ __forceinline__
float activate(ActivationType type,float x,float a) {
    switch (type) {
        case Sigmoid:
            return 1 / (1 + expf(-x));
            

        case Tanh:
            return tanhf(x);
            
        case ReLU:
            return fmaxf(0,x);
    
        case LeakyReLU:
            return fmaxf(a * x,x);
            
        case PReLU:
            return fmaxf(a * x,x);
            
        case ELU:
            return (x >= 0) ? x : a * (expf(x) - 1);
        
        case SELU:
            return 1.0507 * ((x >= 0) ? x : 1.67326 * (expf(x) - 1));
            
        case Softplus:
            return logf(1 + expf(x));
            
        case Swish:
            return x / (1 + expf(-x));
            
        case Mish:
            return x * tanhf(logf(1 + expf(x)));
            
        case HardSigmoid:
            return fmaxf(0,fminf(1,0.2 * x + 0.5));

        case HardTanh:
            return fmaxf(-1,fminf(1,x));

        case HardSwish:
            return x * fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f));
        
        case Softsign:
            return x / (1 + fabsf(x));
            
        case Gaussian:
            return expf(-(x*x));

        case ArcTan:
            return atanf(x);
            
        case Cube:
            return x * x * x;
            
        case Linear:
            return x;
        
        case ReLU6:
            return fminf(fmaxf(0,x),6);
            
        case LogSigmoid:
            return -logf(1 + expf(-x));
            
        default:
            return x;
            

    }
}
