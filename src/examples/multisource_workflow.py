import numpy as np
import sys
from pathlib import Path

try:
    from src.preprocessing.geophysical_features import build_multisource_feature_stack
    from src.preprocessing.multiscale import append_multiscale_channels
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.preprocessing.geophysical_features import build_multisource_feature_stack
    from src.preprocessing.multiscale import append_multiscale_channels


def main():
    # Synthetic example grids (replace with DEM, gravity, magnetic aligned rasters)
    h, w = 256, 256
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")

    dem = 1200 + 200 * np.sin(4 * xx) * np.cos(3 * yy)
    gravity = -30 + 8 * np.exp(-((xx + 0.2) ** 2 + (yy - 0.1) ** 2) / 0.15)
    magnetic = 120 + 15 * np.sin(6 * xx + 2 * yy)

    stack, channel_names, feature_dict = build_multisource_feature_stack(
        dem=dem,
        gravity=gravity,
        magnetic=magnetic,
        dx=1.0,
        dy=1.0,
        regional_sigma=10.0,
        use_harmonica_derivatives=False,
    )

    # Add multi-scale channels for selected geophysical attributes
    with_multiscale = append_multiscale_channels(
        feature_dict,
        target_keys=["dem", "gravity_residual", "magnetic_residual", "gravity_tilt", "magnetic_tilt"],
        scales=(1, 2, 4),
    )

    fused_names = list(with_multiscale.keys())
    fused_stack = np.stack([with_multiscale[name] for name in fused_names], axis=-1)

    print("Base stack shape:", stack.shape)
    print("Base channels:", len(channel_names))
    print("Multiscale stack shape:", fused_stack.shape)
    print("Multiscale channels:", len(fused_names))

    # Optional uncertainty demo if torch is available
    try:
        import torch
        import torch.nn as nn

        try:
            from src.training.uncertainty import mc_dropout_predict
        except ModuleNotFoundError:
            ROOT = Path(__file__).resolve().parents[2]
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from src.training.uncertainty import mc_dropout_predict

        class TinyDropoutSeg(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout2d(0.25),
                    nn.Conv2d(32, 1, kernel_size=1),
                )

            def forward(self, x):
                return self.net(x)

        x = torch.tensor(fused_stack.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        model = TinyDropoutSeg(in_channels=x.shape[1])
        uncertainty = mc_dropout_predict(model, x, n_samples=20, activation="sigmoid")

        print("Uncertainty maps computed:", {k: tuple(v.shape) for k, v in uncertainty.items()})
    except ImportError:
        print("Torch is not installed in this environment. Skipping MC Dropout demo.")


if __name__ == "__main__":
    main()
