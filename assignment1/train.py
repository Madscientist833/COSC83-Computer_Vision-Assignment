import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Import your dataloader and model
from dataloader import get_dataloader
from srcnn import SuperResolutionCNN

# Metrics
from metrics import calculate_psnr, calculate_ssim, fast_psnr, fast_ssim

#10%
def train(config):
    """
    Train the SuperResolution model
    
    Args:
        config (dict): Configuration parameters
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create directory for saving model and results
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    # Create the model
    model = SuperResolutionCNN(
        scale_factor=config['scale_factor'],
        num_channels=3,
        num_features=config['num_features'],
        num_blocks=config['num_blocks']
    ).to(device)
    
    # Load checkpoint if continuing training
    start_epoch = 0
    if config['resume'] and os.path.exists(config['resume']):
        checkpoint = torch.load(config['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
    
    # Loss function
    criterion = nn.L1Loss() 
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_decay_step'],
        gamma=config['lr_decay_gamma']
    )
    
    # Create dataloaders
    train_dataloader = get_dataloader(
        hr_dir=config['train_dir'],
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        fixed_scale=config['scale_factor'],
        downsample_methods=config['downsample_methods'],
        num_workers=config['num_workers']
    )
    
    val_dataloader = get_dataloader(
        hr_dir=config['val_dir'],
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        fixed_scale=config['scale_factor'],
        downsample_methods=['bicubic'],  # Use only bicubic for validation (faster)
        num_workers=config['num_workers']
    )
    
    # Training statistics
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    best_psnr = 0.0
    
    # TODO: Implement the training loop (10%)
    # Your implementation should handle the following:
    #
    # 1. Create a loop that iterates for the specified number of epochs
    #    - Start from start_epoch (in case training is resumed)
    #    - Continue until config['num_epochs']
    #
    # 2. For each epoch, implement the training phase:
    #    - Set the model to training mode
    #    - Create a progress bar using tqdm
    #    - Iterate through the training batches
    #    - Move data to the device
    #    - Perform forward pass, loss calculation, backward pass, and optimization
    #    - Track and display the loss
    #
    # 3. Update the learning rate using the scheduler after each epoch
    #
    # 4. Implement validation at specified intervals:
    #    - Check if validation should be performed based on config['validation_interval']
    #    - Set the model to evaluation mode
    #    - Iterate through validation batches with torch.no_grad()
    #    - Calculate validation loss, PSNR, and SSIM
    #    - Save sample images showing low-res input, super-resolution output, and high-res target
    #
    # 5. Save checkpoints:
    #    - Save the best model based on PSNR
    #    - Save regular checkpoints based on config['save_every']
    #
    # 6. Track and print training statistics
    #
    # Refer to the docstrings of other functions and the provided configuration
    # to understand the expected behavior and parameters.
    
    # Your code here
    
    latest_psnr = 0.0

    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress_bar:
            if isinstance(batch, dict):
                lr, hr = batch['lr'], batch['hr']
            else:
                lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': epoch_loss / (progress_bar.n + 1)})
        
        train_losses.append(epoch_loss / len(train_dataloader))
        scheduler.step()
        
        # Validation
        if (epoch + 1) % config['validation_interval'] == 0:
            model.eval()
            val_loss = 0.0
            psnr_values = []
            ssim_values = []
            
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    if i >= config['val_batch_limit']:
                        break
                    if isinstance(batch, dict):
                        lr, hr = batch['lr'], batch['hr']
                    else:
                        lr, hr = batch
                    
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    
                    loss = criterion(sr, hr)
                    val_loss += loss.item()
                    
                    # Calculate PSNR and SSIM
                    psnr = fast_psnr(sr.cpu(), hr.cpu())
                    ssim = fast_ssim(sr.cpu(), hr.cpu())
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)
                    
                    # Save sample images for the first batch of validation
                    if i == 0:
                        save_image(lr.cpu(), os.path.join(config['sample_dir'], f'epoch_{epoch+1}_lr.png'))
                        save_image(sr.cpu(), os.path.join(config['sample_dir'], f'epoch_{epoch+1}_sr.png'))
                        save_image(hr.cpu(), os.path.join(config['sample_dir'], f'epoch_{epoch+1}_hr.png'))
            
            avg_val_loss = val_loss / min(len(val_dataloader), config['val_batch_limit'])
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            latest_psnr = avg_psnr
            
            val_losses.append(avg_val_loss)
            val_psnrs.append(avg_psnr)
            val_ssims.append(avg_ssim)
            
            print(f"Validation - Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': best_psnr
                }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
                print(f"New best model saved with PSNR: {best_psnr:.2f} dB")
        
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': latest_psnr
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved for epoch {epoch+1}")
    
    # Plot training history (after training loop completes)
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    # Filter out None values for validation
    val_epochs = [i for i, v in enumerate(val_losses) if v is not None]
    val_loss_values = [v for v in val_losses if v is not None]
    plt.plot(val_epochs, val_loss_values, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot PSNR
    plt.subplot(1, 3, 2)
    val_psnr_values = [v for v in val_psnrs if v is not None]
    plt.plot(val_epochs, val_psnr_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR on Validation Set')
    
    # Plot SSIM
    plt.subplot(1, 3, 3)
    val_ssim_values = [v for v in val_ssims if v is not None]
    plt.plot(val_epochs, val_ssim_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM on Validation Set')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_history.png'))
    
    print("Training completed!")


if __name__ == "__main__":
    # Configuration
    config = {
        # Model parameters
        'scale_factor': 4,              # Upscaling factor
        'num_features': 64,             # Number of feature channels
        'num_blocks': 16,               # Number of residual blocks
        
        # Data parameters
        'train_dir': 'DIV2K_train_HR',  # Training data directory
        'val_dir': 'DIV2K_valid_HR',    # Validation data directory
        'patch_size': 128,              # Size of high-resolution patches
        'downsample_methods': ['bicubic', 'bilinear', 'nearest', 'lanczos'],
        
        # Training parameters
        'batch_size': 16,                # Batch size
        'num_epochs': 20,               # Total number of epochs
        'learning_rate': 1e-4,           # Initial learning rate
        'lr_decay_step': 30,             # Epoch interval to decay LR
        'lr_decay_gamma': 0.5,           # Multiplicative factor of learning rate decay
        'num_workers': 4,                # Number of data loading workers
        'validation_interval': 1,        # Epoch interval to perform validation (set to 5 for faster training)
        'val_batch_limit': 10,           # Maximum number of validation batches to process
        
        # Checkpoint parameters
        'checkpoint_dir': 'checkpoints', # Directory to save checkpoints
        'sample_dir': 'samples',         # Directory to save sample images
        'save_every': 5,                 # Save checkpoint every N epochs
        'resume': None,                  # Path to checkpoint to resume from
    }
    
    train(config)