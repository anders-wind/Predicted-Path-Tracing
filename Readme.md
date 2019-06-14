# Path-Predicter

The goal of this project is to make a supervised machine learning model which can predict pixel values of a none-complete path tracing image.


## Background

Path tracing can generate stunning renders with physically accurate properties. Unfortunately the complexity of path tracing makes it near impossible to achieve real time performance. With the increasing focus on machine learning and the new push towards ray tracing graphics with the RTX line of Nvidia GPUS, I hope to combine these two fields, to make a POC of predicting renders.

## Modules

### Renderer (C++)

A path tracing renderer with support for live rendering (the texture is showed and updated always), as well as outputting images at certain times/rendering completion percentadges
The renderer should also support having a postprocessing step where a model predicts the values of the render.

Part of the renderer will be a dataset creater will run the renderer and store the resulting renders at different times. 

### Training (Python)

The training module should contain a pytorch ML model as well as training definition and testing and evaluation. The goal is to output a model.



## Litterature

- http://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf
- https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf
- https://arxiv.org/pdf/1612.03144.pdf
