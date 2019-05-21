# Path-Predicter

The goal of this project is to make a supervised machine learning model which can predict pixel values of a none-complete path tracing image.


## Background

Path tracing can generate stunning renders with physically accurate properties. Unfortunately the complexity of path tracing makes it near impossible to achieve real time performance. With the increasing focus on machine learning and the new push towards ray tracing with the RTX line of Nvidia Graphics Cards, I hope to combine these two, to make a POC of predicting renders.

## Modules

### Renderer

A path tracing renderer with support for live rendering (the texture is showed and updated always), as well as outputting images at certain times/rendering completion percentadges
The renderer should also support having a postprocessing step where a model predicts the values of the render.

### Dataset Creater
The dataset creater will run the renderer and store the resulting renders at different times. 

### Training

The training module should contain a pytorch ML model as well as training definition and testing and evaluation. The goal is to output a model.

