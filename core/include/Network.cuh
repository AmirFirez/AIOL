#pragma once
#include <vector>
#include "Layer.h"
#include "Config.h"
#include "Data.h"


class Network {
    private:

    std::vector<Layer> layers;                          // Layers in the network
    Config& config;                                      // Config of the layer
    Data& data;                                          // Data of the layer 
    cublasHandle_t handle;                              // Cublas handle of the layer


    void initialize_layers(){ // initializing layers
        layers.reserve(config.layer_count); // Reserving the size of the layers so it dosn't resize everytime a new layer is added
        for (size_t i = 0;i < config.layer_count;i++) { // Loops in the layers
            layers.emplace_back(config , i); // Give the layer the config and its index to the layer constructor
        }
    }

    public:

        Network(Config& cfg , Data& dta) // Assign the data given in the constructor to the data in the network
            : config(cfg) , data(dta) // Assigning the data
        {
            cublasCreate(&handle);                              // Makes the cublas handle in the refrence 
        }

        void build() { // Build the network by initializing the layers
            layers.clear();
            initialize_layers(); // initializie layers
        }

        std::vector<float> forward() { // forward for all layers
            float* inputs_to_feed = nullptr;
            cudaMalloc((void**)&inputs_to_feed , data.inputs.size() * sizeof(float));
            cudaMemcpy(inputs_to_feed , data.inputs.data() , data.inputs.size() * sizeof(float) , cudaMemcpyHostToDevice);
            float* first_input = inputs_to_feed;

            for (size_t i = 0;i < config.layer_count;i++) { // Loop in layers to do forward
                inputs_to_feed = layers[i].forward(inputs_to_feed,handle); // Does the forward in the layer with the inputs and cublas handle
            }

            
            cudaFree(first_input);
            return layers.back().get_output();

        }

        ~Network() {
            cublasDestroy(handle);
        }

};

