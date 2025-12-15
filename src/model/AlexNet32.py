import torch
import torch.nn as nn
import numpy as np
from skimage.util import view_as_windows
import torch.nn.init as init

class AlexNet32(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class MiniAlexNet32(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def sliding_window_pixel_center_inference(image_tensor, model, window_size=32, stride=1, device='cpu'):
    """
    image_tensor: torch.Tensor [C,H,W], flotante, no normalizado
    model: red entrenada (devuelve 1 valor por ventana, sigmoid)
    window_size: tamaño de la ventana (32)
    stride: stride del sliding window
    device: 'cpu' o 'cuda'
    
    Devuelve:
    prob_map: np.array (H,W) con probabilidades
    """
    model.eval()
    c, h, w = image_tensor.shape
    center = window_size // 2

    # Inicializar salida y contador de solapamiento
    output = torch.zeros((h, w), dtype=torch.float32)
    count = torch.zeros((h, w), dtype=torch.float32)

    # Convertir a numpy para view_as_windows
    image_np = image_tensor.cpu().numpy()
    img_win = view_as_windows(image_np, (c, window_size, window_size), step=stride)
    num_windows_h, num_windows_w = img_win.shape[1], img_win.shape[2]
    img_win = img_win.reshape(-1, c, window_size, window_size)

    # Convertir a tensor para inferencia
    img_win_tensor = torch.from_numpy(img_win).float().to(device)

    # Inferencia
    with torch.no_grad():
        probs = model(img_win_tensor)  # shape [N,1] o [N,1,1]
        probs = probs.view(-1).cpu()   # aplanar a [N]

    # Reconstrucción colocando predicción en pixel central
    idx = 0
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            output[i + center, j + center] += probs[idx]
            count[i + center, j + center] += 1
            idx += 1

    # Promediar solapamientos
    count[count == 0] = 1.0  # evitar división por cero
    prob_map = (output / count).numpy()

    return prob_map
