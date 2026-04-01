#pragma once
#include <vector>
#include <cublas_v2.h>
#include "Config.h"
#include "../../kernels/include/BiasActivation.cuh"
#include "../../Utils/WeightInit.h"
#include <random>

class Layer {

    private:
        // Data living CPU (Host)
        std::vector<float> h_weights;                 // Weights 1d array (out_features * in_features)
        std::vector<float> h_biases;                  // Biases (out_features)
        size_t in_features;                           // In features
        size_t out_features;                          // Out features
        float alpha;                                  // Alpha
        float beta;                                   // Beta
        float a;                                      // Used in some functions like ("PReLU" , "ELU" , etc)
        int batch_size = 1;                           // Batch size defult 1 (not using batch system)
        ActivationType activation;

        // Data living in GPU VRAM (Device)
        float* d_weights = nullptr;                 // Weights 1d array (out_features * in_features)
        float* d_biases = nullptr;                  // Biases (out_features)
        float* d_act = nullptr;                     // Activated outputs of the layer 1d array (batch_size * out_features)
        float* d_pre_act = nullptr;                 // Pre activated outputs of the layer 1d array (batch_size * out_features)
        float* d_delta = nullptr;                   // Delta (batch_size * out_features)



    public:

        Layer(Config&cfg , size_t layer_idx) {
            in_features = (layer_idx == 0) ? cfg.input_size : cfg.neurons_per_layer[layer_idx-1];
            out_features = cfg.neurons_per_layer[layer_idx];
            batch_size = cfg.batch;
            alpha = cfg.alpha;
            beta = cfg.beta;
            a = cfg.a;
            activation = cfg.activation_func_per_layer[layer_idx];
            h_weights.resize(out_features * in_features);
            h_biases.resize(out_features,0);
            
            InitWeights(cfg.WeightsType,h_weights,in_features,out_features);


            cudaMalloc((void**)&d_weights , (in_features * out_features) * sizeof(float));
            cudaMalloc((void**)&d_biases , out_features* sizeof(float));
            cudaMalloc((void**)&d_act , (batch_size * out_features) * sizeof(float));
            cudaMalloc((void**)&d_pre_act , (batch_size * out_features) * sizeof(float));
            cudaMalloc((void**)&d_delta , (batch_size * out_features) * sizeof(float));

            cudaMemcpy(d_weights , h_weights.data() , (in_features * out_features)  * sizeof(float) , cudaMemcpyHostToDevice);
            cudaMemcpy(d_biases, h_biases.data() , out_features * sizeof(float) , cudaMemcpyHostToDevice);



        }
        ~Layer() {
            cudaFree(d_weights);
            cudaFree(d_biases);
            cudaFree(d_act);
            cudaFree(d_pre_act);
            cudaFree(d_delta);
        }


        void forward(const float* inputs,cublasHandle_t handle) {

            cublasSgemm(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                out_features,
                batch_size,
                in_features,
                &alpha,
                d_weights,
                in_features,
                inputs,
                in_features,
                &beta,
                d_pre_act,
                out_features
            );

            dim3 threads(256);
            dim3 blocks((out_features + 255) / 256, batch_size);

            forward_bias_activation<<<blocks,threads>>>(d_biases,d_pre_act,d_act,out_features,activation,a);

        }

};