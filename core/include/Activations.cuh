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
float activate(ActivationType type,float x,float a = 0.01f) {
    switch (type) {
        case Sigmoid:
            return 1.0f / (1.0f + expf(-x));
            

        case Tanh:
            return tanhf(x);
            
        case ReLU:
            return fmaxf(0.0f,x);
    
        case LeakyReLU:
            return fmaxf(a * x,x);
            
        case PReLU:
            return fmaxf(a * x,x);
            
        case ELU:
            return (x >= 0.0f) ? x : a * (expf(x) - 1.0f);
        
        case SELU:
            return 1.0507f * ((x >= 0.0f) ? x : 1.67326f * (expf(x) - 1.0f));
            
        case Softplus:
            return logf(1.0f + expf(x));
            
        case Swish:
            return x / (1.0f + expf(-x));
            
        case Mish:
            return x * tanhf(logf(1.0f + expf(x)));
            
        case HardSigmoid:
            return fmaxf(0.0f,fminf(1.0f,0.2f * x + 0.5f));

        case HardTanh:
            return fmaxf(-1.0f,fminf(1.0f,x));

        case HardSwish:
            return x * fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f));
        
        case Softsign:
            return x / (1.0f + fabsf(x));
            
        case Gaussian:
            return expf(-(x*x));

        case ArcTan:
            return atanf(x);
            
        case Cube:
            return x * x * x;
            
        case Linear:
            return x;
        
        case ReLU6:
            return fminf(fmaxf(0.0f,x),6.0f);
            
        case LogSigmoid:
            return -logf(1.0f + expf(-x));
            
        default:
            return x;
            

    }
}

__device__ __forceinline__
float activate_derivative(ActivationType type,float x,float a = 0.01f) {
    switch (type) {
        case Sigmoid: {
            float s = activate(Sigmoid,x);
            return s * (1.0f-s);
        }

        case Tanh: { 
            float t = activate(Tanh,x);
            return 1.0f - t*t;
        }

        case ReLU:
            return (x > 0.0f) ? 1.0f : 0.0f;

        case LeakyReLU:
            return (x > 0.0f) ? 1.0f : 0.01f;

        case PReLU:
            return (x > 0.0f) ? 1.0f : a;

        case ELU:
            return (x > 0.0f) ? 1.0f : a*expf(x);

        case SELU:
            return (x > 0.0f) ? 1.0507f : 1.0507f * 1.67326f * expf(x);

        case Softplus:
            return activate(Sigmoid,x);

        case Swish: {
            float s = activate(Sigmoid,x);
            return s + x * s * (1.0f-s);
        }

        case Mish: {
            float sp = logf(1.0f + expf(x));
            float t = activate(Tanh,sp);
            return t + x * activate(Sigmoid,x) * (1.0f-t*t);
        }

        case HardSigmoid:
            return ((-2.5f < x) && (x < 2.5f)) ? 0.2f : 0.0f;

        case HardTanh:
            return ((-1.0f < x) && (x < 1.0f)) ? 1.0f : 0.0f;

        case HardSwish:
            return (x <= -3.0f) ? 0.0f : ((x >= 3.0f) ? 1.0f : (x/3.0f + 0.5f)); 

        case Softsign: {
            float v = (1.0f + fabsf(x));
            return 1.0f / (v*v);
        }

        case Gaussian:
            return -2.0f * x * expf(x * -x);

        case ArcTan:
            return 1.0f / (1.0f + x*x);

        case Cube:
            return 3.0f * (x * x);

        case Linear:
            return x;

        case ReLU6:
            return ((0.0f < x) && (x < 6.0f)) ? 1.0f : 0.0f;

        case LogSigmoid:
            return activate(Sigmoid,x);

        default:
            return x;
    }   
}