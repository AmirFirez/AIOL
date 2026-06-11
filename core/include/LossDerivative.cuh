#pragma once
#include <cmath>


enum LossDerivativeType { // Types of loss derivatives
    MSE,
    MAE,
    Huber,
    BCE,
    BCEWithLogits,
    CE,
    SCE,
    KLDiv,
    Hinge,
    SquaredHinge,
    LogCosh,
    Poisson,
    ExpLoss,
    Quantile,
    SmoothL1
};


__device__ __forceinline__
float sign(float x) { // GPU sign function 
    return (x > 0.0f) ? 1.0f : ((x == 0.0f) ? 0.0f : -1.0f);
}

__device__ __forceinline__
float LossDerivative(LossDerivativeType type , float y , float y_hat , float q , float delta) { // y is the labels here (the expected output) and y_hat is the prediction of the nerual network and q is a Quantile variable and delta is a Huber variable 
    switch (type) {
        case MSE: {
            return 2.0f * (y_hat - y);
        }
        case MAE: {
            return sign(y_hat - y);
        }
        case Huber: {
            return (fabsf(y_hat - y) <= delta) ? (y_hat - y) : (delta * sign(y_hat - y));
        }
        case BCE: {
            y_hat = fmaxf(fminf(y_hat, 1.0f - 1e-7f), 1e-7f);
            return (y_hat - y) / (y_hat * (1.0f - y_hat));
        }
        case BCEWithLogits: {
            return (1.0f / (1.0f + expf(-y_hat))) - y;
        }
        case CE: {
            return y_hat - y;
        }
        case SCE: {
            return y_hat - y;
        }
        case KLDiv: {
            y_hat = fmaxf(fminf(y_hat, 1.0f - 1e-7f), 1e-7f);
            return -(y / y_hat);
        }
        case Hinge: {
            return ((y * y_hat) < 1.0f) ? -y : 0.0f;
        }
        case SquaredHinge: {
            return ((y_hat * y) < 1.0f) ? ((-2.0f * y) * (1.0f - y * y_hat)): 0.0f;
        }
        case LogCosh: {
            return tanhf(y_hat - y);
        }
        case Poisson: {
            y_hat = fmaxf(fminf(y_hat, 1.0f - 1e-7f), 1e-7f);
            return 1.0f - (y / y_hat);
        }
        case ExpLoss: {
            return -y * expf(-y * y_hat);
        }
        case Quantile: {
            return (y - y_hat > 0.0f) ? -q : (1.0f - q);
        }
        case SmoothL1: {
            return (fabsf(y_hat - y) < 1.0f) ? (y_hat - y) : sign(y_hat - y);
        }
        default : {
            return 0;
        }
    }
}