# **Benchmark Results**

This benchmark evaluates the performance improvements achieved by integrating **Tritonserver-rs** into a video processing pipeline.

## **Setup**  

The benchmark uses models to:
1. Detect objects in a video frame.
2. Perform regression tasks on the detected objects.

The test video resolution was **Full HD (1920x1080)**. The **model** used for inference was lightweight, but the **frame flow** in the pipeline was very high, simulating a demanding real-world application where processing efficiency is critical.

## **Methods Compared**  

The models were executed using four different methods:  
1. **Dedicated Triton Server**: Requests sent via **gRPC**.  
2. **Python Triton Library**: Direct **CUDA memory** transfer.  
3. **Tritonserver-rs**: Leveraging local execution.  
4. **DeepStream SDK**: Optimized for video pipelines.  

Tests were conducted across various GPUs, and the table below shows the average frames per second (FPS) processed by each method:  

| **GPU**        | **Triton (gRPC)** | **Triton (Shared Memory)** | **Tritonserver-rs** | **DeepStream SDK** |  
|:--------------:|:-----------------:|:--------------------------:|:-------------:|:------------------:|  
| Tesla T4       | 70                | 105                        | 200           | 320                |  
| RTX 3090 Ti    | 80                | 140                        | 360           | 450                |  
| A10            | 75                | 115                        | 270           | 330                |  
| A100           | 80                | 130                        | 330           | 400                |  

---

## **Key Observations**  

1. **Performance Gains**:  
   - **Tritonserver-rs** outperformed the dedicated Triton Server by a factor of **3–4x** compared to gRPC-based communication.  
   - Compared to the Python Triton library with shared memory, **Tritonserver-rs** delivered **2x the performance**.  

2. **Comparison with DeepStream**:  
   While **DeepStream SDK** achieves the highest FPS due to its specialization in video processing, it comes at the cost of flexibility and broader model support. **Tritonserver-rs** offers a balanced trade-off, combining significant performance improvements with flexibility for various use cases.

3. **Hardware Agnosticism**:  
   Model execution on different GPUs required no additional configuration. This demonstrates the adaptability and ease of deployment of **Tritonserver-rs** across a wide range of hardware.  