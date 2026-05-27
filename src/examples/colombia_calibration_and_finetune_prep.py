#!/usr/bin/env python
"""
Colombia Calibration & Fine-tuning Preparation.

Este script:
1. Carga el checkpoint entrenado en Atrato
2. Hace inferencia sobre todos los tiles de Colombia
3. Calcula curvas PR y calibración
4. Selecciona umbrales óptimos por objetivo
5. Prepara datos para fine-tuning
"""

import sys
import os
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model.Org_UNet import UNet


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate checkpoint on Colombia and compute PR/calibration metrics")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / 'data' / 'atrato_training' / 'checkpoints' / 'atrato_unet.pt'),
        help="Checkpoint path (.pt/.pth) to evaluate",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output folder name (e.g., 'v2')",
    )
    return parser.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])

# Paths
CHECKPOINT_PATH = Path(ARGS.checkpoint)
COL_STACK_DIR = PROJECT_ROOT / 'data' / 'Colombia' / 'training' / 'stacks'
COL_MASK_DIR = PROJECT_ROOT / 'data' / 'Colombia' / 'training' / 'masks'
out_dir_name = 'tmp_evaluation' if not ARGS.output_suffix else f"tmp_evaluation_{ARGS.output_suffix}"
OUTPUT_DIR = PROJECT_ROOT / out_dir_name
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Dispositivo: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Stacks Colombia: {COL_STACK_DIR}")
print(f"Masks Colombia: {COL_MASK_DIR}")
print()


def infer_tile(model, stack_path, device='cpu'):
    """Carga un tile, normaliza y hace inferencia. Retorna predicción (probabilidad)."""
    stack = np.load(stack_path).astype(np.float32)  # (H, W, 5)
    # Normalizar por canal
    for i in range(stack.shape[2]):
        band = stack[:, :, i]
        valid = band[np.isfinite(band)]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            std_val = np.std(valid)
            if std_val > 0:
                stack[:, :, i] = (band - mean_val) / std_val
    # NaN → 0
    stack = np.nan_to_num(stack, 0)
    # (H, W, 5) → (1, 5, H, W)
    x = torch.from_numpy(np.transpose(stack, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)  # (1, 1, H, W)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
    return probs


# ============================================================================
# 1. Cargar modelo
# ============================================================================
print("="*70)
print("1. CARGANDO MODELO")
print("="*70)

if not CHECKPOINT_PATH.exists():
    print(f"ERROR: checkpoint no existe en {CHECKPOINT_PATH}")
    sys.exit(1)

model = UNet(n_classes=1, input_channels=5)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
print("✓ Modelo cargado y en modo eval")
print()


# ============================================================================
# 2. Inferencia sobre Colombia
# ============================================================================
print("="*70)
print("2. INFERENCIA SOBRE COLOMBIA")
print("="*70)

col_stack_files = sorted(glob.glob(str(COL_STACK_DIR / '*.npy')))
col_mask_files = sorted(glob.glob(str(COL_MASK_DIR / '*.npy')))

print(f"Tiles encontrados: {len(col_stack_files)} stacks, {len(col_mask_files)} masks")

col_predictions = []
col_ground_truths = []
col_tile_names = []

print("Ejecutando inferencia...")
for i, (sp, mp) in enumerate(zip(col_stack_files, col_mask_files)):
    pred_prob = infer_tile(model, sp, device=DEVICE)
    gt_mask = np.load(mp).astype(np.float32)
    
    col_predictions.append(pred_prob)
    col_ground_truths.append(gt_mask)
    col_tile_names.append(Path(sp).stem)
    
    if (i + 1) % 10 == 0 or (i + 1) == len(col_stack_files):
        print(f"  {i+1}/{len(col_stack_files)}")

print(f"✓ Inferencia completada: {len(col_predictions)} tiles")
print()

# Concatenar todas las predicciones
all_probs = np.concatenate([p.ravel() for p in col_predictions])
all_gts = np.concatenate([g.ravel() for g in col_ground_truths])

print(f"Total píxeles: {len(all_gts):,}")
print(f"Píxeles positivos GT: {np.sum(all_gts > 0.5):,} ({100*np.mean(all_gts>0.5):.3f}%)")
print(f"Píxeles predichos (prob > 0.5): {np.sum(all_probs > 0.5):,} ({100*np.mean(all_probs>0.5):.3f}%)")
print()


# ============================================================================
# 3. Funciones para métricas
# ============================================================================

def compute_pr_curve(gt, probs, n_thresholds=200):
    """Calcula curva PR para diferentes umbrales."""
    thresholds = np.linspace(0, 1, n_thresholds)
    precisions, recalls, f1s, ious, accuracies = [], [], [], [], []
    
    gt_binary = (gt > 0.5).astype(np.float32)
    for thresh in thresholds:
        pred = (probs >= thresh).astype(np.float32)
        tp = np.sum(gt_binary * pred)
        fp = np.sum((1 - gt_binary) * pred)
        fn = np.sum(gt_binary * (1 - pred))
        tn = np.sum((1 - gt_binary) * (1 - pred))
        
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        iou = tp / max(tp + fp + fn, 1e-8)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ious.append(iou)
        accuracies.append(acc)
    
    return thresholds, np.array(precisions), np.array(recalls), np.array(f1s), np.array(ious), np.array(accuracies)


def reliability_curve(gt_mask, pred_prob, bins=10):
    """Calcula curva de calibración."""
    gt = (gt_mask > 0.5).astype(np.float32).ravel()
    prob = pred_prob.astype(np.float32).ravel()
    edges = np.linspace(0, 1, bins + 1)
    centers, avg_conf, avg_acc, counts = [], [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1:
            sel = (prob >= left) & (prob <= right)
        else:
            sel = (prob >= left) & (prob < right)
        if not np.any(sel):
            continue
        centers.append((left + right) / 2)
        avg_conf.append(float(np.mean(prob[sel])))
        avg_acc.append(float(np.mean(gt[sel])))
        counts.append(int(np.sum(sel)))
    return np.array(centers), np.array(avg_conf), np.array(avg_acc), np.array(counts)


# ============================================================================
# 4. Calcular curvas y métricas
# ============================================================================
print("="*70)
print("3. CURVAS PR Y CALIBRACIÓN")
print("="*70)

thresholds, precisions, recalls, f1s, ious, accuracies = compute_pr_curve(all_gts, all_probs, n_thresholds=200)

# Baseline
gt_binary = (all_gts > 0.5).astype(np.float32)
pred_baseline = (all_probs >= 0.5).astype(np.float32)
tp_b = np.sum(gt_binary * pred_baseline)
fp_b = np.sum((1 - gt_binary) * pred_baseline)
fn_b = np.sum(gt_binary * (1 - pred_baseline))
tn_b = np.sum((1 - gt_binary) * (1 - pred_baseline))

print("[BASELINE - umbral 0.5]")
print(f"  Accuracy:  {(tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b):.4f}")
print(f"  Precision: {tp_b / max(tp_b + fp_b, 1e-8):.4f}")
print(f"  Recall:    {tp_b / max(tp_b + fn_b, 1e-8):.4f}")
print(f"  F1:        {2 * (tp_b / max(tp_b + fp_b, 1e-8)) * (tp_b / max(tp_b + fn_b, 1e-8)) / max((tp_b / max(tp_b + fp_b, 1e-8)) + (tp_b / max(tp_b + fn_b, 1e-8)), 1e-8):.4f}")
print(f"  IoU:       {tp_b / max(tp_b + fp_b + fn_b, 1e-8):.4f}")
print()


# ============================================================================
# 5. Seleccionar umbrales por objetivo
# ============================================================================
print("="*70)
print("4. SELECCIÓN DE UMBRAL OPERATIVO POR OBJETIVO")
print("="*70)

targets = [0.70, 0.75, 0.80, 0.85, 0.90]
threshold_results = {}

for target_recall in targets:
    # Encuentra el threshold con recall >= target_recall que maximiza F1
    valid_idx = np.where(recalls >= target_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1s[valid_idx])]
        t_opt = thresholds[best_idx]
        rec_opt = recalls[best_idx]
        prec_opt = precisions[best_idx]
        f1_opt = f1s[best_idx]
        iou_opt = ious[best_idx]
        
        threshold_results[target_recall] = {
            'threshold': float(t_opt),
            'precision': float(prec_opt),
            'recall': float(rec_opt),
            'f1': float(f1_opt),
            'iou': float(iou_opt),
        }
        
        print(f"Target Recall >= {target_recall:.0%}:")
        print(f"  Threshold: {t_opt:.4f}")
        print(f"  Precision: {prec_opt:.4f}")
        print(f"  Recall:    {rec_opt:.4f}")
        print(f"  F1:        {f1_opt:.4f}")
        print(f"  IoU:       {iou_opt:.4f}")
    else:
        print(f"Target Recall >= {target_recall:.0%}: NO ALCANZABLE")
    print()

# Mejores F1 e IoU globales
best_f1_idx = np.argmax(f1s)
best_iou_idx = np.argmax(ious)

print("[ÓPTIMOS GLOBALES]")
print(f"Mejor F1:  {f1s[best_f1_idx]:.4f} @ threshold {thresholds[best_f1_idx]:.4f}")
print(f"Mejor IoU: {ious[best_iou_idx]:.4f} @ threshold {thresholds[best_iou_idx]:.4f}")
print()


# ============================================================================
# 6. Visualizaciones
# ============================================================================
print("="*70)
print("5. GENERANDO VISUALIZACIONES")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# PR curve
axes[0].plot(recalls, precisions, 'b-', linewidth=2, label='PR')
for target_recall in [0.70, 0.80, 0.90]:
    if target_recall in threshold_results:
        r = threshold_results[target_recall]['recall']
        p = threshold_results[target_recall]['precision']
        axes[0].plot(r, p, 'o', markersize=8)
        axes[0].text(r, p, f'  {target_recall:.0%}', fontsize=9)
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Curva Precision-Recall')
axes[0].grid(alpha=0.3)
axes[0].legend()

# F1 vs threshold
axes[1].plot(thresholds, f1s, 'g-', linewidth=2)
axes[1].plot(thresholds[best_f1_idx], f1s[best_f1_idx], 'ro', markersize=8, 
             label=f'Max F1={f1s[best_f1_idx]:.3f}')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('F1')
axes[1].set_title('F1 vs Threshold')
axes[1].grid(alpha=0.3)
axes[1].legend()

# IoU vs threshold
axes[2].plot(thresholds, ious, 'purple', linewidth=2)
axes[2].plot(thresholds[best_iou_idx], ious[best_iou_idx], 'ro', markersize=8,
             label=f'Max IoU={ious[best_iou_idx]:.3f}')
axes[2].set_xlabel('Threshold')
axes[2].set_ylabel('IoU')
axes[2].set_title('IoU vs Threshold')
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.suptitle('Selección de umbral operativo – Colombia', fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / 'pr_curves.png', dpi=100, bbox_inches='tight')
print("✓ Guardado: pr_curves.png")
plt.close()


# Curva de calibración
centers, avg_conf, avg_acc, counts = reliability_curve(all_gts, all_probs, bins=10)
ece = np.mean(np.abs(avg_conf - avg_acc))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5, label='Perfecta (y=x)')
ax.plot(avg_conf, avg_acc, marker='o', markersize=8, linewidth=2, label='Observada')
for x, y, count in zip(avg_conf, avg_acc, counts):
    ax.text(x, y+0.03, str(count), fontsize=8, ha='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Confianza media (predicción)')
ax.set_ylabel('Frecuencia positiva observada (GT)')
ax.set_title(f'Curva de calibración – Colombia (ECE={ece:.4f})')
ax.grid(alpha=0.3)
ax.legend()
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'calibration_curve.png', dpi=100, bbox_inches='tight')
print("✓ Guardado: calibration_curve.png")
plt.close()


# ============================================================================
# 7. Preparar datos para fine-tuning
# ============================================================================
print()
print("="*70)
print("6. PREPARACIÓN PARA FINE-TUNING")
print("="*70)

# Split 70-30
rng_split = np.random.default_rng(42)
n_col_tiles = len(col_stack_files)
n_train_ft = int(0.70 * n_col_tiles)

indices_all = np.arange(n_col_tiles)
rng_split.shuffle(indices_all)

indices_train_ft = indices_all[:n_train_ft]
indices_val_ft = indices_all[n_train_ft:]

print(f"Total Colombia tiles: {n_col_tiles}")
print(f"Fine-tuning train: {len(indices_train_ft)} tiles")
print(f"Fine-tuning validation: {len(indices_val_ft)} tiles")
print()

# Guardar índices
ft_split = {
    'train': [col_tile_names[i] for i in indices_train_ft],
    'val': [col_tile_names[i] for i in indices_val_ft],
}

split_file = OUTPUT_DIR / 'colombia_finetune_split.json'
with open(split_file, 'w') as f:
    json.dump(ft_split, f, indent=2)
print(f"✓ Guardado: {split_file.name}")
print()


# ============================================================================
# 8. Guardar resultados en JSON
# ============================================================================
print("="*70)
print("7. GUARDANDO RESULTADOS")
print("="*70)

results = {
    'checkpoint': str(CHECKPOINT_PATH),
    'dataset': {
        'stack_dir': str(COL_STACK_DIR),
        'mask_dir': str(COL_MASK_DIR),
        'n_tiles': n_col_tiles,
        'pixels_total': int(all_gts.size),
        'gt_positive_fraction': float(np.mean(all_gts > 0.5)),
    },
    'baseline_metrics_0_5': {
        'threshold': 0.5,
        'accuracy': float((tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b)),
        'precision': float(tp_b / max(tp_b + fp_b, 1e-8)),
        'recall': float(tp_b / max(tp_b + fn_b, 1e-8)),
        'f1': float((2 * (tp_b / max(tp_b + fp_b, 1e-8)) * (tp_b / max(tp_b + fn_b, 1e-8))) / 
                    max((tp_b / max(tp_b + fp_b, 1e-8)) + (tp_b / max(tp_b + fn_b, 1e-8)), 1e-8)),
        'iou': float(tp_b / max(tp_b + fp_b + fn_b, 1e-8)),
    },
    'optimal_thresholds': threshold_results,
    'calibration_error': {
        'ece': float(ece),
    },
    'finetune_split': {
        'n_train': len(indices_train_ft),
        'n_val': len(indices_val_ft),
    },
}

results_file = OUTPUT_DIR / 'evaluation_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Guardado: {results_file.name}")
print()

print("="*70)
print("RESUMEN FINAL")
print("="*70)
print(json.dumps(results, indent=2))
print()
print(f"Todos los archivos guardados en: {OUTPUT_DIR}")
print()
