# Detección semiautomática de lineamientos geológicos con aprendizaje automático

Este repositorio contiene el desarrollo del proyecto de investigación enfocado en la **detección semiautomática de estructuras geológicas** mediante algoritmos de aprendizaje profundo. Se parte del análisis de datos geofísicos (gravedad, magnéticos) y de elevación (DEM) para identificar fallas, fracturas, contactos litológicos y otros lineamientos estructurales clave.

---

### Objetivo general

Desarrollar algoritmos de aprendizaje automático para detectar y vectorizar lineamientos de manera semiautomática a partir de datos geofísicos y de elevación.
---

### Tecnologías utilizadas

| Categoría | Herramienta / Biblioteca |
|---|---|
| Lenguaje | Python 3.11 |
| Deep Learning | PyTorch |
| Geofísica | [Harmonica](https://www.fatiando.org/harmonica/), [Verde](https://www.fatiando.org/verde/) |
| Geodatos | xarray, rasterio, geopandas, GDAL |
| Procesamiento numérico | NumPy, SciPy |
| Visión por computador | scikit-image |
| Análisis / XAI | SHAP |
| Visualización | Matplotlib |
| Entorno | Conda / venv |

---

### Estructura del repositorio

```
lineament_detection/
├── data/
│   ├── Colombia/               # Datos y etiquetas del Mapa Geológico de Colombia (SGC)
│   │   └── labels/sgc_mgc2023/ # Fallas, pliegues y tectónica (GeoJSON)
│   └── OpenData/               # Datos satelitales abiertos (Sentinel-2, geofísica global)
│       └── non_colombia/       # AOIs: California, Turquía, Islandia
├── src/
│   ├── model/                  # Arquitecturas de redes neuronales
│   │   ├── Org_UNet.py         # U-Net estándar con n canales de entrada
│   │   ├── LNetCodDec.py       # Red encoder-decoder ligera
│   │   ├── AlexNet32.py        # AlexNet adaptado para segmentación
│   │   ├── cGAN.py             # GAN condicional para generación de máscaras
│   │   └── RotateNet.py        # Red con invarianza rotacional
│   ├── preprocessing/          # Construcción de features y datasets
│   │   ├── geophysical_features.py   # TGA, tilt, señal analítica, Gabor, texturas, curvatura
│   │   ├── multiscale.py             # Pirámides multiescala de features
│   │   ├── multisource_dataset.py    # Dataset PyTorch para stacks multicanal
│   │   ├── prepare_multisource_data.py
│   │   ├── rasterized_training_data.py
│   │   ├── faultMapping.py
│   │   └── bsds500.py                # Integración con el benchmark de detección de bordes BSDS500
│   ├── training/               # Entrenamiento, métricas e incertidumbre
│   │   ├── train.py
│   │   ├── multisource_pipeline.py   # Pipeline completo multifuente
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── uncertainty.py            # MC Dropout (media, std, entropía, inf. mutua)
│   │   └── uncertainty_geotiff.py    # Inferencia con incertidumbre sobre GeoTIFFs
│   ├── plotting/               # Visualización de resultados
│   │   ├── plot_images.py
│   │   └── plot_metrics.py
│   └── examples/               # Scripts de ejemplo y flujos de trabajo completos
│       ├── multisource_workflow.py
│       ├── run_full_multisource_pipeline.py
│       ├── train_multisource_unet.py
│       ├── train_and_test_colombia.py
│       ├── colombia_calibration_and_finetune_prep.py
│       ├── infer_multisource_with_uncertainty.py
│       ├── infer_prestacked_raster_with_uncertainty.py
│       ├── prepare_training_from_prestacked_raster_and_vectors.py
│       ├── download_open_data_and_connect_pipeline.py
│       ├── download_global_geophysics.py
│       ├── rasterize_gem_faults.py
│       ├── generate_presentation_figures.py
│       └── shap.py                   # Análisis de importancia de features (SHAP)
├── notebooks/                  # Notebooks de análisis y visualización
├── models/                     # Modelos entrenados (.pth)
├── outputs/                    # Resultados de predicciones y vectorización
└── docs/                       # Documentación técnica y científica
```

---

### Características principales

#### 1. Extracción de características geofísicas (`geophysical_features.py`)
A partir de un DEM y grillas de gravedad/magnéticos, se computan:
- **Derivadas del DEM**: pendiente, curvatura laplaciana, curvatura gaussiana, curvatura media, perfil y plan.
- **Amplitud del gradiente total (TGA)** y **ángulo de inclinación (tilt)** para gravedad y magnéticos.
- **Señal analítica (AS)**, **HGM**, **theta map** y **SVD** (segunda derivada vertical).
- **Separación regional/residual** por suavizado gaussiano.
- **Derivadas direccionales** en acimuts geográficos (NE-SW, NW-SE).
- **Filtros Gabor** orientados para detección de lineamientos.
- **Texturas espaciales** (varianza local, entropía local).

#### 2. Stack multicanal y multiescala
`build_multisource_feature_stack` fusiona todas las features. 
`append_multiscale_channels` permite capturar estructuras regionales y locales.

#### 3. Arquitecturas de segmentación
- **U-Net** 
- **LNetCodDec**
- **AlexNet32**
- **cGAN**
- **RotateNet**

#### 4. Estimación de incertidumbre (MC Dropout)
`mc_dropout_predict` realiza *n* pasadas estocásticas sobre el modelo en modo evaluación, devolviendo: **media**, **desviación estándar**, **entropía**, **entropía esperada** e **información mutua** (incertidumbre epistémica).

#### 5. Datos de entrenamiento y validación
- **Colombia**
- **Global**: Sentinel-2 L2A sobre la falla de San Andrés (California), falla Norte de Anatolia (Turquía) y rift de Þingvellir (Islandia).

---

### Instalación

```bash
# Crear entorno e instalar dependencias
conda env create -f notebooks/environment.yml
conda activate lineament_detection
```

Para el flujo de trabajo completo (descarga de datos → preprocesamiento → entrenamiento → inferencia con incertidumbre), ver [`src/examples/run_full_multisource_pipeline.py`](src/examples/run_full_multisource_pipeline.py).

---

### Licencia

Este repositorio hace parte de un proyecto de ciencia abierta.

---

### Agradecimientos

Este trabajo hace parte de la línea de investigación del **Grupo de Mecánica Aplicada** de la Universidad EAFIT, financiado por **Minciencias** y la **Universidad EAFIT**.

---

### Contacto

| Nombre | Rol | Perfil |
|---|---|---|
| **Nicolas Guarín-Zapata** | Profesor Asistente – Universidad EAFIT | [CvLAC](https://scienti.minciencias.gov.co/cvlac/visualizador/generarCurriculoCv.do?cod_rh=0000994316) |
| **Maria Alejandra Botero Botero** | Joven Investigadora – Universidad EAFIT | [CvLAC](https://scienti.minciencias.gov.co/cvlac/visualizador/generarCurriculoCv.do?cod_rh=0001826916) |
| **Mauricio Felipe Revelo Aristizábal** | Colaborador | — |
| **Julián Andrés Solano Zapardiel** | Colaborador | — |

---
---

# Semi-automatic Geological Lineament Detection with Machine Learning

This repository contains the research project for the **semi-automatic detection of geological structures** using deep learning algorithms. The methodology relies on geophysical data (gravity, magnetics) and digital elevation models (DEM) to identify faults, fractures, lithological contacts, and other key structural lineaments.

---

### General Objective

Develop machine learning algorithms to detect and vectorize lineaments semi-automatically using geophysical and elevation data. 

### Technologies

| Category | Tool / Library |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch |
| Geophysics | [Harmonica](https://www.fatiando.org/harmonica/), [Verde](https://www.fatiando.org/verde/) |
| Geodata | xarray, rasterio, geopandas, GDAL |
| Numerics | NumPy, SciPy |
| Computer Vision | scikit-image |
| Analysis / XAI | SHAP |
| Visualisation | Matplotlib |
| Environment | Conda / venv |

---

### Repository Structure

```
lineament_detection/
├── data/
│   ├── Colombia/               # Colombian Geological Map labels (SGC)
│   │   └── labels/sgc_mgc2023/ # Faults, folds, tectonics (GeoJSON)
│   └── OpenData/               # Open satellite/geophysical data
│       └── non_colombia/       # AOIs: California, Turkey, Iceland (Sentinel-2)
├── src/
│   ├── model/                  # Neural-network architectures
│   │   ├── Org_UNet.py         # Standard U-Net (arbitrary input channels)
│   │   ├── LNetCodDec.py       # Lightweight encoder-decoder
│   │   ├── AlexNet32.py        # AlexNet adapted for dense prediction
│   │   ├── cGAN.py             # Conditional GAN for mask generation
│   │   └── RotateNet.py        # Rotation-invariant network
│   ├── preprocessing/          # Feature engineering and datasets
│   │   ├── geophysical_features.py   # TGA, tilt, analytic signal, Gabor, texture, curvature
│   │   ├── multiscale.py             # Multi-scale feature pyramids
│   │   ├── multisource_dataset.py    # PyTorch dataset for multichannel .npy stacks
│   │   ├── prepare_multisource_data.py
│   │   ├── rasterized_training_data.py
│   │   ├── faultMapping.py
│   │   └── bsds500.py                # BSDS500 edge-detection benchmark integration
│   ├── training/               # Training, metrics & uncertainty
│   │   ├── train.py
│   │   ├── multisource_pipeline.py   # End-to-end multi-source pipeline
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── uncertainty.py            # MC Dropout (mean, std, entropy, mutual info)
│   │   └── uncertainty_geotiff.py    # Uncertainty inference on GeoTIFFs
│   ├── plotting/               # Result visualisation
│   │   ├── plot_images.py
│   │   └── plot_metrics.py
│   └── examples/               # End-to-end workflow scripts
│       ├── multisource_workflow.py
│       ├── run_full_multisource_pipeline.py
│       ├── train_multisource_unet.py
│       ├── train_and_test_colombia.py
│       ├── colombia_calibration_and_finetune_prep.py
│       ├── infer_multisource_with_uncertainty.py
│       ├── infer_prestacked_raster_with_uncertainty.py
│       ├── prepare_training_from_prestacked_raster_and_vectors.py
│       ├── download_open_data_and_connect_pipeline.py
│       ├── download_global_geophysics.py
│       ├── rasterize_gem_faults.py
│       ├── generate_presentation_figures.py
│       └── shap.py                   # Feature-importance analysis (SHAP)
├── notebooks/                  # Exploratory analysis and visualisation notebooks
├── models/                     # Trained model checkpoints (.pth)
├── outputs/                    # Prediction and vectorisation outputs
└── docs/                       # Technical and scientific documentation
```

---

### Key Features

#### 1. Geophysical Feature Extraction (`geophysical_features.py`)
From a DEM and gravity/magnetic grids, the pipeline computes:
- **DEM derivatives**: slope (rad & deg), Laplacian curvature, Gaussian, mean, profile, and plan curvatures.
- **Total Gradient Amplitude (TGA)** and **tilt angle** for gravity and magnetics.
- **Analytic signal (AS)**, **Horizontal Gradient Magnitude (HGM)**, **theta map**, and **Second Vertical Derivative (SVD)**.
- **Regional / residual separation** via Gaussian smoothing.
- **Directional derivatives** at geographic azimuths (NE-SW 45°, NW-SE 135°).
- **Gabor filter** for orientation-selective lineament detection.
- **Spatial textures**: local variance (contrast) and local entropy.

#### 2. Multichannel & Multi-scale Stack
`build_multisource_feature_stack` fuses all features. 
`append_multiscale_channels` allows to capture both regional and local structures.

#### 3. Segmentation Architectures
- **U-Net**
- **LNetCodDec**
- **AlexNet32**
- **cGAN**
- **RotateNet**

#### 4. Uncertainty Estimation (MC Dropout)
`mc_dropout_predict` performs *n* stochastic forward passes in evaluation mode, returning: **mean prediction**, **standard deviation**, **entropy**, **expected entropy**, and **mutual information** (epistemic uncertainty).

#### 5. Training & Validation Data
- **Colombia**
- **Global benchmarks**: Sentinel-2 L2A over the San Andreas Fault (California), North Anatolian Fault (Turkey), and Þingvellir rift (Iceland).

---

### Installation

```bash
# Create environment and install dependencies
conda env create -f notebooks/environment.yml
conda activate lineament_detection
```

For the complete end-to-end workflow (data download → preprocessing → training → uncertainty-aware inference), see [`src/examples/run_full_multisource_pipeline.py`](src/examples/run_full_multisource_pipeline.py).

---

### License

This repository is part of an open-science project.

---

### Acknowledgements

This work is part of the research line of the **Applied Mechanics Group** (Grupo de Mecánica Aplicada) at Universidad EAFIT, funded by **Minciencias** and **Universidad EAFIT**.

---

### Contact

| Name | Role | Profile |
|---|---|---|
| **Nicolas Guarín-Zapata** | Assistant Professor – Universidad EAFIT | [CvLAC](https://scienti.minciencias.gov.co/cvlac/visualizador/generarCurriculoCv.do?cod_rh=0000994316) |
| **Maria Alejandra Botero Botero** | Young Researcher – Universidad EAFIT | [CvLAC](https://scienti.minciencias.gov.co/cvlac/visualizador/generarCurriculoCv.do?cod_rh=0001826916) |
| **Mauricio Felipe Revelo Aristizábal** | Collaborator | — |
| **Julián Andrés Solano Zapardiel** | Collaborator | — |
