# Python-to-Triton LLM: Automated Code Translation

## About Project

This project, `python-to-triton-llm`, is an advanced end-to-end machine learning initiative focused on developing a specialized LLM capable of automatically translating Python function bodies into highly optimized Triton kernels. It addresses the critical need to bridge the gap between Python's development agility and Triton's unparalleled performance for GPU-accelerated computations. Built upon a foundation of synthetic data.
## Architecture and Metrics



* **Synthetic Data Pipeline:**
    * A custom pipeline generates and processes over **2,000 unique synthetic Python-Triton code pairs**.
    * Each pair undergoes rigorous **automated validation** via dedicated test scripts, ensuring data integrity and correctness before inclusion in the training set.
    * 
* **Transformer LLM Architecture:**
    * At its core is a **50M-parameter Transformer-based LLM**, custom-implemented in PyTorch.
    * The model (`MinimalLLM`) features a multi-layered decoder-only architecture, incorporating advanced components such as **Multi-Head Attention with Rotary Position Embeddings (RoPE)** and a Feed-Forward Network.

* **Training & Optimization Metrics:**
    * The training regimen employs **mixed-precision (AMP)** for accelerated computation and reduced memory footprint.
    * **Gradient accumulation** is utilized to achieve a **4x effective batch size increase** (e.g., from 24 to 96), enabling more stable training with larger effective batch sizes without requiring excessive GPU memory.
    * A novel **Muon optimizer**, a hybrid momentum-orthogonalized approach, is implemented to enhance model convergence and stability.


## Technical Features

* **Custom Transformer LLM Implementation:** From-scratch PyTorch implementation of a decoder-only Transformer, providing granular control over architecture and training.
* **Automated Synthetic Data Generation & Validation:** A sophisticated system for creating diverse Python-Triton code examples and programmatically verifying their correctness, ensuring a high-quality training corpus.
* **Advanced Training Techniques:** Integration of state-of-the-art optimization strategies including Automatic Mixed Precision (AMP), gradient accumulation, and a custom Muon optimizer for efficient and effective model training.
* **Python-to-Triton Code Generation:** The trained LLM is capable of taking a Python function as input and generating a corresponding, functionally equivalent Triton kernel.
* **Interactive Demonstration:** Includes an interactive command-line interface for real-time testing and demonstration of the model's code generation capabilities.
* **Modular and Extensible Design:** The codebase is structured into distinct modules for data processing, model definition, and training logic, facilitating future enhancements and scalability.
