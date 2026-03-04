import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, fused, ir, vis):
        # Intensity loss (max of ir and vis)
        target_intensity = torch.max(ir, vis)
        loss_intensity = self.l1(fused, target_intensity)
        
        # Gradient loss
        grad_fused_x, grad_fused_y = self.gradient(fused)
        grad_ir_x, grad_ir_y = self.gradient(ir)
        grad_vis_x, grad_vis_y = self.gradient(vis)
        
        # Get max gradient target
        target_grad_x = torch.max(torch.abs(grad_ir_x), torch.abs(grad_vis_x))
        target_grad_y = torch.max(torch.abs(grad_ir_y), torch.abs(grad_vis_y))
        
        loss_grad = self.l1(torch.abs(grad_fused_x), target_grad_x) + \
                    self.l1(torch.abs(grad_fused_y), target_grad_y)
                    
        loss_total = loss_intensity + 5.0 * loss_grad
        return loss_total
        
    def gradient(self, x):
        # Using a simple gradient filter (Sobel)
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(x, kernel_x, padding=1)
        grad_y = F.conv2d(x, kernel_y, padding=1)
        return grad_x, grad_y
