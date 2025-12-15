import torch
from collections import defaultdict
import time
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import training.losses as losses
from training.metrics import print_metrics

def train_model(
    model, 
    optimizer, 
    scheduler, 
    dataloaders, 
    num_epochs=25, 
    device='cpu'
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            # Crear barra de progreso
            progress_bar = tqdm(
                enumerate(dataloaders[phase]), 
                total=len(dataloaders[phase]),
                desc=f'{phase.capitalize()}',
                ncols=100
            )

            for batch_idx, (inputs, labels) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = losses.calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                epoch_samples += inputs.size(0)
                
                # Actualizar información en la barra
                progress_bar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'avg_loss': f'{metrics["loss"]/epoch_samples:.4f}'
                })

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val':
                val_losses.append(epoch_loss)
            else:
                train_losses.append(epoch_loss)
                scheduler.step()

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss

def train_model_sliding_window(
    model,
    optimizer,
    scheduler,
    dataloaders,
    num_epochs=25,
    device='cpu',
    save_best=True
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-'*20)
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR:", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            pbar = tqdm(dataloaders[phase], desc=f'{phase} phase')
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = losses.calc_loss_sw(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

                pbar.set_postfix({'loss': loss.item()})

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val':
                val_losses.append(epoch_loss)

                if save_best and epoch_loss < best_loss:
                    print("Saving BEST validation model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            else:
                train_losses.append(epoch_loss)
                scheduler.step()

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:.4f}'.format(best_loss))

    if save_best:
        print("Loading best validation model")
        model.load_state_dict(best_model_wts)
    else:
        print("Keeping LAST epoch model (not the best val model)")

    return model, train_losses, val_losses, best_loss

def train_gan(
    generator, 
    discriminator, 
    dataloaders,
    g_optimizer,
    d_optimizer,
    num_epochs=100, 
    device="cuda"
):
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    best_val_loss = float("inf")
    best_gen_state = None

    train_g_losses = []
    train_d_losses = []
    val_g_losses = []
    val_d_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                generator.train()
                discriminator.train()
            else:
                generator.eval()
                discriminator.eval()

            g_running_loss = 0.0
            d_running_loss = 0.0
            batches = 0

            progress_bar = tqdm(
                enumerate(dataloaders[phase]),
                total=len(dataloaders[phase]),
                desc=f'{phase.upper()}',
                ncols=120
            )

            for batch_idx, (img, label) in progress_bar:
                img, label = img.to(device), label.to(device)
                batches += 1

                # Train Discriminator
                with torch.set_grad_enabled(phase == "train"):
                    fake_label = generator(img).detach()
                
                    if phase == "train":
                        noise_real = torch.randn_like(label) * 0.05
                        noise_fake = torch.randn_like(fake_label) * 0.05
                        label_noisy = (label + noise_real).clamp(0, 1)
                        fake_label_noisy = (fake_label + noise_fake).clamp(0, 1)
                    else:
                        label_noisy = label
                        fake_label_noisy = fake_label
                    
                    d_optimizer.zero_grad()

                    real_pred = discriminator(img, label_noisy)
                    fake_pred = discriminator(img, fake_label_noisy)

                    # Label smoothing
                    real_loss = criterion_gan(real_pred, torch.ones_like(real_pred) * 0.9)
                    fake_loss = criterion_gan(fake_pred, torch.ones_like(fake_pred) * 0.1)
                    d_loss = (real_loss + fake_loss) * 0.5

                    if phase == "train":
                        d_loss.backward()
                        d_optimizer.step()

                # Train Generator
                with torch.set_grad_enabled(phase == "train"):
                    fake_label = generator(img)
                    g_optimizer.zero_grad()

                    pred_fake = discriminator(img, fake_label)
                    gan_loss = criterion_gan(pred_fake, torch.ones_like(pred_fake))
                    
                    # L1 loss
                    l1_loss = criterion_l1(torch.sigmoid(fake_label), label)
                    
                    # BCE
                    pos_weight = losses.calculate_pos_weight(label)
                    criterion_bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    bce_loss = criterion_bce_weighted(fake_label, label)
                    
                    # Combined loss
                    g_total = gan_loss * 1.0 + l1_loss * 50 + bce_loss * 50
                    
                    if phase == "train":
                        g_total.backward()
                        g_optimizer.step()

                g_running_loss += g_total.item()
                d_running_loss += d_loss.item()

                progress_bar.set_postfix({
                    'G_total': f'{g_total.item():.4f}',
                    'GAN': f'{gan_loss.item():.4f}',
                    'D_loss': f'{d_loss.item():.4f}',
                    'D_real_mean': f'{real_pred.mean().item():.3f}',
                    'D_fake_mean': f'{fake_pred.mean().item():.3f}',
                })

            g_epoch_loss = g_running_loss / batches
            d_epoch_loss = d_running_loss / batches

            print(f"{phase.upper()}  G_loss={g_epoch_loss:.4f}  D_loss={d_epoch_loss:.4f}")

            if phase == "train":
                train_g_losses.append(g_epoch_loss)
                train_d_losses.append(d_epoch_loss)
            else:
                val_g_losses.append(g_epoch_loss)
                val_d_losses.append(d_epoch_loss)

                if g_epoch_loss < best_val_loss:
                    best_val_loss = g_epoch_loss
                    best_gen_state = copy.deepcopy(generator.state_dict())
                    print("Saving best generator model...")

    generator.load_state_dict(best_gen_state)
    return generator, train_g_losses, train_d_losses, val_g_losses, val_d_losses

def train_model_pos_weight(
    model,
    optimizer,
    scheduler,
    dataloaders,
    num_epochs=25,
    device='cpu',
    loss_weights={'bce': 1.0, 'dice': 1.0, 'focal': 0.5}
):
    """
    Entrena un modelo utilizando una combinación ponderada de BCE, Dice y Focal Loss,
    calculando dinámicamente el pos_weight para BCE en cada batch.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'\n=== Epoch {epoch+1}/{num_epochs} ===')
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                model.eval()

            running_loss = 0.0
            running_bce = 0.0
            running_dice = 0.0
            running_focal = 0.0
            epoch_samples = 0

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f'{phase.upper()}',
                ncols=140
            )

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    # Calcular pos_weight para BCE
                    pos_weight = losses.calculate_pos_weight(labels)
                    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    
                    # Calcular losses individuales
                    bce_loss = criterion_bce(outputs, labels)
                    dice_l = losses.dice_loss_and_sigmoid(outputs, labels)
                    focal_l = losses.focal_loss_with_logits(outputs, labels)
                    
                    # Loss total ponderado
                    total_loss = (
                        loss_weights['bce'] * bce_loss +
                        loss_weights['dice'] * dice_l +
                        loss_weights['focal'] * focal_l
                    )

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # Acumular losses
                running_loss += total_loss.item() * batch_size
                running_bce += bce_loss.item() * batch_size
                running_dice += dice_l.item() * batch_size
                running_focal += focal_l.item() * batch_size
                epoch_samples += batch_size

                # Actualizar barra de progreso
                progress_bar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'bce': f'{bce_loss.item():.4f}',
                    'dice': f'{dice_l.item():.4f}',
                    'focal': f'{focal_l.item():.4f}',
                    'avg': f'{running_loss/epoch_samples:.4f}'
                })

            # Calcular losses promedio de la época
            epoch_loss = running_loss / epoch_samples
            epoch_bce = running_bce / epoch_samples
            epoch_dice = running_dice / epoch_samples
            epoch_focal = running_focal / epoch_samples

            print(f'{phase.upper()} - Total: {epoch_loss:.4f} | BCE: {epoch_bce:.4f} | Dice: {epoch_dice:.4f} | Focal: {epoch_focal:.4f}')

            if phase == 'val':
                val_losses.append(epoch_loss)
                
                # Guardar mejor modelo
                if epoch_loss < best_loss:
                    print(f'✓ Mejor validación: {epoch_loss:.4f} (anterior: {best_loss:.4f})')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                train_losses.append(epoch_loss)
                scheduler.step()

        time_elapsed = time.time() - since
        print(f'Tiempo: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    print(f'\n{"="*50}')
    print(f'Mejor validation loss: {best_loss:.4f}')
    print(f'{"="*50}')

    # Cargar los mejores pesos
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss