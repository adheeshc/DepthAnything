import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2 as DAV2Model


class DepthAnythingV2:
    """
    models: 'vits', 'vitb', 'vitl'
    """

    def __init__(self, model_name="vitb", device="cuda", use_fp16=True):
        self.device = device
        self.use_fp16 = use_fp16
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

        config = model_configs[model_name]
        self.model = DAV2Model(**config).to(device)
        checkpoints = {
            "vits": "./checkpoints/depth_anything_v2_vits.pth",
            "vitb": "./checkpoints/depth_anything_v2_vitb.pth",
            "vitl": "./checkpoints/depth_anything_v2_vitl.pth",
        }
        self.model.load_state_dict(
            torch.load(checkpoints[model_name], map_location="cpu")
        )
        self.model.eval()
        if use_fp16:
            self.model.half()

    @torch.no_grad()
    def infer(self, image):
        """Predict depth from image"""

        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        image_resized = cv2.resize(
            image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        image_resized = image_resized / 255.0

        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        if self.use_fp16:
            image_tensor = image_tensor.half()
        else:
            image_tensor = image_tensor.float()

        depth = self.model(image_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True
        ).squeeze()
        depth_np = depth.cpu().float().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        return depth_np

    def infer_metric_depth(
        self, image, method="least_squares", gt_depth=None, camera_height=None
    ):
        """
        Convert relative depth to metric depth using various methods

        Args:
            image: input RGB image
            method: 'least_squares' (requires gt_depth), 'camera_height' (requires camera_height),
                   or 'assumed_median' (uses fixed assumption)
            gt_depth: optional ground truth depth for least squares alignment
            camera_height: optional camera height above ground plane in meters

        Returns:
            depth_metric: metric depth in meters (h, w)
        """
        depth_relative = self.infer(image)

        if method == "least_squares" and gt_depth is not None:
            # Use least squares to find optimal scale
            valid_mask = (gt_depth > 0) & (gt_depth < 10)
            pred_valid = depth_relative[valid_mask].flatten()
            gt_valid = gt_depth[valid_mask].flatten()

            # Solve: gt = scale * pred using least squares
            scale = np.median(gt_valid / (pred_valid + 1e-6))
            depth_metric = scale * depth_relative

        elif method == "camera_height" and camera_height is not None:
            # Estimate scale using camera height and ground plane assumption
            h, w = depth_relative.shape
            ground_region = depth_relative[int(h * 0.8) :, :]

            # Assume highest relative depth values are ground plane
            ground_depth_relative = np.percentile(ground_region, 95)

            # Scale so that ground plane is at camera_height distance
            scale = camera_height / (ground_depth_relative + 1e-6)
            depth_metric = scale * depth_relative

        else:
            # Assumed median depth (fallback method)
            median_depth_m = 3.0
            scale = median_depth_m / (np.median(depth_relative) + 1e-6)
            depth_metric = scale * depth_relative

        return depth_metric

    @staticmethod
    def compute_depth_metrics(pred_depth, gt_depth, mask=None):
        """Compute depth estimation metrics between prediction and ground truth"""
        if mask is None:
            mask = (gt_depth > 0) & (gt_depth < 10)

        mask = mask & (pred_depth > 0) & np.isfinite(pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        if len(pred_depth) == 0 or len(gt_depth) == 0:
            return {
                "abs_rel": float("inf"),
                "sq_rel": float("inf"),
                "rmse": float("inf"),
                "rmse_log": float("inf"),
                "a1": 0.0,
                "a2": 0.0,
                "a3": 0.0,
            }

        scale = np.median(gt_depth) / (np.median(pred_depth) + 1e-8)
        pred_depth = pred_depth * scale

        pred_depth = np.maximum(pred_depth, 1e-8)
        gt_depth = np.maximum(gt_depth, 1e-8)

        # Threshold metrics
        thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25**2).mean()
        a3 = (thresh < 1.25**3).mean()

        # Error metrics
        abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
        sq_rel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)
        rmse = np.sqrt(np.mean((gt_depth - pred_depth) ** 2))
        rmse_log = np.sqrt(np.mean((np.log(gt_depth) - np.log(pred_depth)) ** 2))

        return {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "a1": a1,
            "a2": a2,
            "a3": a3,
        }


def get_RGBD_pairs(dataset_path, max_time_diff=0.02):
    """Associate RGB and depth images from TUM dataset based on timestamps"""
    rgb_txt = os.path.join(dataset_path, "rgb.txt")
    depth_txt = os.path.join(dataset_path, "depth.txt")

    # Read RGB timestamps
    rgb_list = []
    with open(rgb_txt, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp = float(parts[0])
                filename = parts[1].split("/")[-1]
                rgb_list.append((timestamp, filename))

    # Read depth timestamps
    depth_list = []
    with open(depth_txt, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp = float(parts[0])
                filename = parts[1].split("/")[-1]
                depth_list.append((timestamp, filename))

    # Associate pairs based on closest timestamps
    associations = []
    for rgb_ts, rgb_file in rgb_list:
        min_diff = float("inf")
        best_match = None

        for depth_ts, depth_file in depth_list:
            time_diff = abs(rgb_ts - depth_ts)
            if time_diff < min_diff:
                min_diff = time_diff
                best_match = (depth_ts, depth_file)

        if best_match and min_diff < max_time_diff:
            associations.append((rgb_file, best_match[1], rgb_ts))

    return associations


def test_depth_anything(use_metric_depth=False):
    """Evaluate depth estimation on entire TUM dataset

    Args:
        use_metric_depth: If True, use least squares method to estimate metric depth
    """
    model = DepthAnythingV2(model_name="vitb", device="cuda")

    dataset_path = "../data/rgbd_dataset_freiburg1_desk"
    rgb_path = os.path.join(dataset_path, "rgb")
    depth_path = os.path.join(dataset_path, "depth")

    associations = get_RGBD_pairs(dataset_path)

    method_str = "with Least Squares Metric Depth" if use_metric_depth else "with Relative Depth"
    print(f"Evaluating on {len(associations)} RGB-D pairs {method_str}\n")

    all_metrics = {
        "abs_rel": [],
        "sq_rel": [],
        "rmse": [],
        "rmse_log": [],
        "a1": [],
        "a2": [],
        "a3": [],
    }

    for rgb_file, depth_file, _ in tqdm(
        associations, desc="Evaluating depth", unit="image"
    ):
        img_path = os.path.join(rgb_path, rgb_file)
        gt_depth_path = os.path.join(depth_path, depth_file)

        image = cv2.imread(img_path)
        gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH)
        gt_depth = gt_depth.astype(np.float32) / 5000.0

        if use_metric_depth:
            pred_depth = model.infer_metric_depth(image, method='least_squares', gt_depth=gt_depth)
        else:
            pred_depth = model.infer(image)

        if gt_depth.shape != pred_depth.shape:
            gt_depth = cv2.resize(gt_depth, (pred_depth.shape[1], pred_depth.shape[0]))

        metrics = model.compute_depth_metrics(pred_depth, gt_depth)
        if np.isfinite(metrics["rmse"]):
            for key in all_metrics.keys():
                all_metrics[key].append(metrics[key])

    results = {}
    for metric_name in all_metrics.keys():
        values = np.array(all_metrics[metric_name])
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        min_val = np.min(values)
        max_val = np.max(values)

        results[metric_name] = {
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
        }

        print(
            f"{metric_name.upper():12s} | Mean: {mean_val:.4f} | Std: {std_val:.4f} | "
            f"Median: {median_val:.4f} | Min: {min_val:.4f} | Max: {max_val:.4f}"
        )

    # save results
    suffix = "_metric" if use_metric_depth else "_relative"
    results_file = f"depth_evaluation_results{suffix}.csv"
    with open(results_file, "w") as f:
        f.write("image_idx,rgb_file,depth_file,abs_rel,sq_rel,rmse,rmse_log,a1,a2,a3\n")
        for idx, (rgb_file, depth_file, _) in enumerate(
            associations[: len(all_metrics["rmse"])]
        ):
            f.write(f"{idx},{rgb_file},{depth_file},")
            f.write(f"{all_metrics['abs_rel'][idx]:.6f},")
            f.write(f"{all_metrics['sq_rel'][idx]:.6f},")
            f.write(f"{all_metrics['rmse'][idx]:.6f},")
            f.write(f"{all_metrics['rmse_log'][idx]:.6f},")
            f.write(f"{all_metrics['a1'][idx]:.6f},")
            f.write(f"{all_metrics['a2'][idx]:.6f},")
            f.write(f"{all_metrics['a3'][idx]:.6f}\n")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Depth Estimation Metrics Distribution", fontsize=16)

    metrics_to_plot = [
        ("abs_rel", "Absolute Relative Error"),
        ("rmse", "RMSE (m)"),
        ("rmse_log", "RMSE Log"),
        ("a1", "δ < 1.25"),
        ("a2", "δ < 1.25²"),
        ("a3", "δ < 1.25³"),
    ]

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = all_metrics[metric_key]
        ax.hist(values, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(values),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(values):.4f}",
        )
        ax.axvline(
            np.median(values),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(values):.4f}",
        )
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Frequency")
        ax.set_title(metric_label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = f"depth_metrics_distribution{suffix}.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Metrics distribution plot saved to {plot_file}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    test_depth_anything(use_metric_depth=True)
