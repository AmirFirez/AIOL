# AIOL

> A custom GPU-accelerated deep learning framework built from scratch using CUDA and cuBLAS.

---

## 🚀 Overview

AIOL is a performance-focused deep learning framework designed to give full control over neural network internals, memory management, and GPU execution.

Unlike many traditional frameworks, AIOL is being developed from the ground up with a focus on:

* Low-level control
* Modular architecture
* GPU-first design

---

## ⚙️ Current Features

* ✅ CUDA-based forward pass
* ✅ cuBLAS accelerated matrix multiplication
* ✅ Fused bias + activation CUDA kernel
* ✅ Layer-based architecture
* ✅ Config-driven model design
* ✅ Multiple activation functions
* ✅ Weight initialization strategies (He, Xavier, etc.)
* ✅ Host ↔ Device memory management

---

## ⚠️ Project Status: Beta

This project is currently in an active early beta stage.

The core forward pipeline is implemented and functional, but the framework is still under active development and subject to major improvements and changes.

---

## ⚠️ Current Limitations

 These limitations will be addressed in upcoming versions.

* ❌ Backpropagation not implemented yet
* ⚠️ Limited error handling (CUDA / cuBLAS calls not fully checked)
* ⚠️ Memory management is currently basic and will be further optimized
* ⚠️ Global RNG
* ⚠️ API is unstable and may change
* ⚠️ Limited batching optimizations

---

## 🧠 Design Philosophy

* Built completely from scratch (no external ML frameworks)
* GPU-first architecture using CUDA
* Modular system (Layer, Config, Kernels, Utils)
* Focus on performance and flexibility
* Learning-oriented but aiming for real-world capability

---

## 🧱 Core Components

### 🔹 Layer

* Manages weights, biases, and GPU memory
* Executes forward pass using cuBLAS
* Applies fused bias + activation kernel

### 🔹 Config

* Defines model architecture and hyperparameters
* Controls activations, initialization, batch size, etc.

### 🔹 Kernels

* Custom CUDA kernels for:

  * Bias addition
  * Activation functions
    Both are fused for performance

### 🔹 Utils

* Weight initialization system

### 🔹 Network

* Initializie layers 
* Connect layers and data and config togther


---


## 🧪 Example (Coming Soon)

Example usage and training pipeline will be added after backpropagation is implemented.

---

## 📌 Notes

This project is actively evolving.
Expect breaking changes as the architecture improves.

---

## 📄 License

Apache 2.0 License

---

## 👨‍💻 Author

Built and maintained as a custom deep learning framework project.
