#pragma once
#include <random>
#include <vector>

enum WeightsType {                  // Cases for every way to intilaize weights
    HE,
    XavierNormal,
    XavierUniform,
    Normal,
    Zero
};

std::random_device rd;
std::mt19937 gen(rd());            // Makes a random generation



void InitWeights(WeightsType type,std::vector<float>& h_weights, float in, float out) {

    in = std::max(in, 1e-8f);      // if in is 0 we make it a very small number so divison doesn't break

    switch (type) {                // Updates weights based on the case we want to use
        case HE: {
            std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / in)); // Makes a normal distribution which works around a range
            for (auto& w : h_weights) {         // Loop over the weights 
                w = dist(gen);                  // Change the weights value
            }
            break;                          // The end of the function
        }

        case XavierNormal: {
            std::normal_distribution<float> dist(0.0f, sqrtf(1.0f / in)); // Makes a normal distribution which works around a range
            for (auto& w : h_weights) {         // Loop over the weights 
                w = dist(gen);                  // Change the weights value
            }
            break;                          // The end of the function
        }

        case XavierUniform: {
            std::uniform_real_distribution<float> dist(-sqrtf(6.0f / (in + out)), sqrtf(6.0f / (in + out))); // Makes a uniform real distribution which works around a hard range 
            for (auto& w : h_weights) {         // Loop over the weights 
                w = dist(gen);                  // Change the weights value
            }
            break;                          // The end of the function
        }

        case Normal: {
            std::normal_distribution<float> dist(0.0f, 0.01f); // Makes a normal distribution which works around a range
            for (auto& w : h_weights) {         // Loop over the weights 
                w = dist(gen);                  // Change the weights value
            }
            break;                          // The end of the function
        }

        case Zero: {
            for (auto& w : h_weights) {         // Loop over the weights 
                w = 0.0f;                  // Change the weights value
            }
            break;                          // The end of the function
        }

        default: {
            for (auto& w : h_weights) {         // Loop over the weights 
                w = 0.01f;                  // Change the weights value
            }
            break;                          // The end of the function
        }

        
    }

}