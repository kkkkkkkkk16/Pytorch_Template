import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from data.M3DF import M3DF
from models.nets.FushionNet import FushionNet
from models.losses.FusionLoss import FusionLoss
import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    train_dataset = M3DF(root=args.dataset_dir, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Model Setup
    model = FushionNet(in_channels=1, out_channels=1, feat_channels=256).to(device)
    
    # Loss and Optimizer
    criterion = FusionLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # ir_Y, vis_Y, vis_Cb, vis_Cr, vis_list_index
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for ir, vis, cb, cr, _ in pbar:
            ir = ir.to(device)
            vis = vis.to(device)
            
            optimizer.zero_grad()
            fused = model(ir, vis)
            
            loss = criterion(fused, ir, vis)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch [{epoch}/{args.epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}")
        
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, f'fusion_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fusion autoencoder")
    parser.add_argument('--dataset_dir', type=str, default='datasets/M3FD', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval for checkpoints')
    
    args = parser.parse_args()
    train(args)
