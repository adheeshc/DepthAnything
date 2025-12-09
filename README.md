# Depth Anything V2 Evaluation

This project implements depth estimation evaluation using the Depth Anything V2 model, comparing relative and metric depth predictions with comprehensive metrics analysis.

## Overview

The evaluation pipeline processes images using Depth Anything V2 to generate depth maps and analyzes the quality of predictions using multiple metrics including RMSE, MAE, Absolute Relative Error, and δ accuracy thresholds.

## Features

- **Dual Depth Prediction Modes**: Supports both relative and metric depth estimation
- **Comprehensive Metrics**: Evaluates predictions using RMSE, MAE, Abs Rel, and δ thresholds (δ < 1.25, 1.25², 1.25³)
- **Visualization**: Generates distribution plots and comparison visualizations
- **CSV Export**: Saves detailed results for further analysis

## Files

- `depth_anythingv2.py` - Main evaluation script
- `depth_evaluation_results.csv` - Relative depth prediction results
- `depth_evaluation_results_metric.csv` - Metric depth prediction results
- `depth_metrics_distribution.png` - Metrics visualization for relative depth
- `depth_metrics_distribution_metric.png` - Metrics visualization for metric depth
- `depth_prediction.png` - Sample depth prediction visualization

## Setup

1. Clone the Depth Anything V2 repository:
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git DepthAnythingV2
```

2. Download model checkpoints and place them in the `checkpoints/` directory

3. Install dependencies:
```bash
pip install torch torchvision opencv-python numpy matplotlib pandas
```

## Usage

Run the evaluation script:
```bash
python depth_anythingv2.py
```

The script will process images and generate:
- Depth predictions
- Evaluation metrics
- Visualization plots
- CSV result files

## Model

This project uses [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), a state-of-the-art monocular depth estimation model.