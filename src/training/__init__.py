__all__ = []

try:
    from .uncertainty import mc_dropout_predict
    __all__.append("mc_dropout_predict")
except ModuleNotFoundError:  # torch is optional for environments without DL deps
    pass

try:
    from .multisource_pipeline import (
        create_multisource_dataloaders,
        predict_stack_with_uncertainty,
    )
    __all__.extend([
        "create_multisource_dataloaders",
        "predict_stack_with_uncertainty",
    ])
except ModuleNotFoundError:  # torch is optional for environments without DL deps
    pass

try:
    from .uncertainty_geotiff import export_uncertainty_maps_to_geotiff
    __all__.append("export_uncertainty_maps_to_geotiff")
except ModuleNotFoundError:  # optional if torch/raster stack isn't available
    pass
