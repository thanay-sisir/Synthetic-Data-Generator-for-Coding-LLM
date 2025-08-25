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


