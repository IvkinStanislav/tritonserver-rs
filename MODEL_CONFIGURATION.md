# **Model Configuration**

The model repository system and `config.pbtxt` format are inherited from the **Triton Inference Server**. For full documentation, refer to the official Triton guide: [Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md).

Below is a brief overview of some powerful Triton configuration options that can simplify your workflow and enhance performance:

- **[Model Warmup](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#onnx-with-tensorrt-optimization-ort-trt)**:  
  If your model requires complex calculations or compilation during initialization, you can configure it to complete these tasks during startup, avoiding delays on the first request.

- **[ONNX ORT Optimization](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#onnx-with-tensorrt-optimization-ort-trt)**:  
  Enables conversion of ONNX models to TensorRT during initialization, often resulting in significant performance improvements.

- **[Dynamic Batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher)**:  
  Configures the server to batch multiple incoming requests together, which can balance server load and improve throughput.

- **[Instance Groups](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)**:  
  Specifies how many instances of a model should be loaded and assigns each instance to a specific GPU or CPU. This setting can optimize performance and enable CPU-only deployments.  
  *(See more details in [CPU-only Run](./DEPLOY.md#cpu-only-mode).)*

- **[Version Policy](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#version-policy)**:  
  By setting the **Version Policy** to `"specific"` in combination with `crate::Options::model_control_mode(EXPLICIT)` and `crate::Server::load_model`, you can load only a single model version instead of all available versions. This is especially useful for:  
  - Managing repositories with multiple versions of a heavy model.  
  - Avoiding loading broken model versions while keeping healthy ones.  

---

This guide provides a starting point for using Triton’s advanced configuration features to streamline your machine learning workflows. For a detailed walkthrough of these options and more, refer to the official [Triton Inference Server documentation](https://github.com/triton-inference-server/server).  
