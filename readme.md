[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
# Temporal Sequence Transformer (TST)

**Non-Autoregressive Time-Series Forecasting and Classification**

## Introduction

The **Temporal Sequence Transformer (TST)** is a non-autoregressive transformer-based architecture designed for efficient time-series forecasting and classification. It leverages a custom attention mechanism, positional encodings (learnable or sinusoidal), and a decoder-only structure to model long-range dependencies in time-series data. This repository implements two models:

- **Non_AR_TST**: For long-term time-series forecasting.
- **Non_AR_TSCT**: For time-series classification (work in progress)

The TST is designed for applications like financial forecasting, traffic analysis, and energy consumption prediction, offering flexibility for both univariate and multivariate datasets.

## Model Overview

The Temporal Sequence Transformer introduces a non-autoregressive approach to time-series tasks, avoiding iterative decoding for faster inference. Key components include:

- **Timestep Embeddings**: Maps timesteps of features to a higher-dimensional embedding space.
- **Positional Encoding**: Sinusoidal or learnable encodings for temporal context.
- **Non-AR Masked Multi-Head Attention**: Captures dependencies across time steps efficiently.
- **Decoder Stack**: Processes input sequences with masked attention for forecasting/classification.
- **Non-Autoregressive Output**: Predicts entire future sequences or class probabilities in one forward pass.

## TST Architecture:

<img src="figures/tstarchitecture.png" alt="TST Architecture" width="600">

## Non-AR Masked Multi-Head Attention:

![TST Architecture](figures/nonarattn.png)

## Results

We benchmarked the `Non_AR_TST` model against the Informer model on long-term forecasting tasks using the **ETTh1**, **ETTh2**, and **ETTm1** datasets. These datasets, part of the Electricity Transformer Temperature (ETT) collection, are designed for evaluating long-term forecasting by providing high-frequency multivariate time-series data from transformer sensors. ETTh1 and ETTh2 capture hourly measurements, while ETTm1 provides minute-level granularity, making them ideal for testing models on extended prediction horizons.

Benchmark results comparing `Non_AR_TST` and Informer are visualized below:

![Benchmark Results](figures/results.png)

*(Note: Ensure `results.png` is available in the `figures/` directory with actual benchmark metrics.)*

## Citation

If you use our work in your research, please cite:

```bibtex
@article{tst2025,
  title={Temporal Sequence Transformer: Non-Autoregressive Time-Series Forecasting and Classification},
  author={Nimbalkar, Paartha},
  journal={TBD},
  year={2025}
}
