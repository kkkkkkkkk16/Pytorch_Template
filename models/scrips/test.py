import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia.utils as KU
import cv2

from data.M3DF import M3DF
from models.nets.FushionNet import FushionNet
from utils.util import YCbCr2RGB
import argparse

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    test_dataset = M3DF(root=args.dataset_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Model Setup
    model = FushionNet(in_channels=1, out_channels=1, feat_channels=256).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with untrained weights.")
        
    model.eval()
    os.makedirs(args.result_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for ir, vis, cb, cr, img_name in pbar:
            ir = ir.to(device)
            vis = vis.to(device)
            cb = cb.to(device)
            cr = cr.to(device)
            
            fused_Y = model(ir, vis)
            
            # Reconstruct RGB image
            fused_RGB = YCbCr2RGB(fused_Y, cb, cr)
            
            # Save image
            img_tensor = fused_RGB.squeeze(0).cpu()
            import numpy as np
            img_np = (KU.tensor_to_image(img_tensor) * 255.0).clip(0, 255).astype(np.uint8)
            
            # Use img_name[0] since batch_size=1 makes it a tuple of size 1
            save_path = os.path.join(args.result_dir, img_name[0])
            cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Fusion autoencoder")
    parser.add_argument('--dataset_dir', type=str, default='datasets/M3FD', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fusion_model_epoch_10.pth', help='Path to model checkpoint')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    test(args)
