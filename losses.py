from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import spherical as S360
import supervision as L
from sphere_xyz import get_uni_sphere_xyz
from layers import *
from planar_origin import generate_planar_depth
from saver import Saver
import numpy
import cv2
torch.manual_seed(10)
torch.cuda.manual_seed(10)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss

class SVS(nn.Module):
    def __init__(self, device, setting):
        super(SVS, self).__init__()
        self.width = 512
        self.height = 256
        self.depth_thresh = 10.0
        self.sgrid = S360.grid.create_spherical_grid(self.width).to(device)
        self.uvgrid = S360.grid.create_image_grid(self.width, self.height).to(device)
        self.device = device
        self.photo_params = L.photometric.PhotometricLossParameters(
            alpha=0.85, l1_estimator='none', ssim_estimator='none',
            ssim_mode='gaussian', std=1.5, window=7
        )
        self.batch_size = setting.batch_size
        self.d2n_nei = setting.d2n_nei
        self.planar_thresh = setting.planar_thresh
        self.lambda_norm_reg = setting.lambda_norm_reg
        self.lambda_planar_reg = setting.lambda_planar_reg
        self.using_normloss = setting.using_normloss
        self.using_disp2seg = setting.using_disp2seg
        self.saver = Saver("./framework_image/")
    
    def forward(self, left_rgb, left_depth_pred, up_depth, up_rgb, vps, epoch, mask=None):
        #传个batch进来先吗？  rgb不要进行数据增强 可能影响灭点检测或者上下图投影转
        disp = torch.cat(
            (
                torch.zeros_like(left_depth_pred),
                S360.derivatives.dtheta_vertical(self.sgrid, left_depth_pred, 0.26)  # args.baseline=0.26
            ),
            dim=1
        )
        up_render_coords = self.uvgrid + disp
        up_render_coords[torch.isnan(up_render_coords)] = 0.0
        up_render_coords[torch.isinf(up_render_coords)] = 0.0
        up_rgb_t, up_mask_t = L.splatting.render(left_rgb, left_depth_pred, \
                                                 up_render_coords, max_depth=self.depth_thresh)

        
        ''' Loss UD '''
        up_cutoff_mask = (up_depth < self.depth_thresh)
        up_mask_t &= ~(up_depth > self.depth_thresh)
        attention_weights = S360.weights.theta_confidence(
            S360.grid.create_spherical_grid(self.width)).to(self.device)
        # attention_weights = torch.ones_like(left_depth)
        photo1_loss = L.photometric.calculate_loss(up_rgb_t, up_rgb, self.photo_params,
                                                   mask=up_cutoff_mask, weights=attention_weights)
        active_loss = torch.tensor(0.0).to(self.device)
        active_loss += photo1_loss * 1.0  #args.photo=1.0

        ''' Loss Prior (3D Smoothness) '''
        left_xyz = S360.cartesian.coords_3d(self.sgrid, left_depth_pred)
        dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)
        guidance_duv = S360.derivatives.dI_duv(left_rgb)
        # attention_weights = torch.zeros_like(left_depth)
        depth1_smooth_loss = L.smoothness.guided_smoothness_loss(
            dI_dxyz, guidance_duv, up_cutoff_mask, (1.0 - attention_weights)
                                                   * up_cutoff_mask.type(attention_weights.dtype)
        )
        
        active_loss += depth1_smooth_loss * 0.1  #args.smooth_reg_w=0.1
        
        '''Mahattan Align'''
        if self.using_normloss and self.training:  # 测refinenet时，先设置为False
            xyz = left_depth_pred.permute(0, 2, 3, 1) * get_uni_sphere_xyz(self.batch_size, self.height, self.width).to(
                self.device)  
            pred_norm = depth2norm(xyz, self.height, self.width, self.d2n_nei)  
            #vps = batch['vps'].to(self.device)
            mmap, mmap_mask, mmap_mask_thresh = compute_mmap(self.batch_size, pred_norm, vps, self.height, self.width, epoch,
                                                             self.d2n_nei)  
            aligned_norm = align_smooth_norm(self.batch_size, mmap, vps, self.height, self.width)
            

            
            '''Co-Planar'''
            if self.using_disp2seg:
                xyz = xyz.permute(0, 3, 1, 2).reshape(self.batch_size, 3, -1).float()
                out_planar = generate_planar_depth(left_rgb, aligned_norm, xyz, self.width, self.height, self.device, self.batch_size,
                                                   self.planar_thresh) 


        ''' Loss Align-normal'''
        if self.using_normloss and self.training:  
            loss_norm_reg = 0.0
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            norm_loss_score = cos(pred_norm, aligned_norm)  
            #normloss_mask = mmap_mask  

            if self.using_disp2seg:
                planar_mask = out_planar['planar_mask']
                normloss_mask = mmap_mask * planar_mask
                if torch.any(torch.isnan(norm_loss_score)):
                    print('warning! nan is norm loss compute! set nan = 1')
                    norm_loss_score = torch.where(torch.isnan(norm_loss_score), torch.full_like(norm_loss_score, 1),
                                                  norm_loss_score)
            else:
                normloss_mask = mmap_mask  
            
            norm_loss = (1 - norm_loss_score).unsqueeze(1) * normloss_mask
            
            loss_norm_reg += torch.mean(norm_loss)
            active_loss += self.lambda_norm_reg * loss_norm_reg
            ''' Loss Planar'''
            if self.using_disp2seg :
                loss_planar_reg = 0.0
                planar_depth = out_planar['planar_depth']
                planar_mask = out_planar['planar_mask']
                pred_depth = left_depth_pred

                assert torch.isnan(pred_depth).sum() == 0, print(pred_depth)

                if torch.any(torch.isnan(planar_depth)):
                    print('warning! nan in planar_depth!')
                    planar_depth = torch.where(torch.isnan(planar_depth), torch.full_like(planar_depth, 0),
                                               planar_depth)
                    pred_depth = torch.where(torch.isnan(planar_depth), torch.full_like(pred_depth, 0), pred_depth)
                planar_loss = torch.abs(pred_depth - planar_depth) * planar_mask
                # planar_loss *= attention_weights
                loss_planar_reg += torch.mean(planar_loss)
                active_loss += self.lambda_planar_reg * loss_planar_reg

        return active_loss
        
class SVS_plane(nn.Module):
    def __init__(self, device, setting):
        super(SVS_plane, self).__init__()
        self.width = 512
        self.height = 256
        self.depth_thresh = 10.0
        self.sgrid = S360.grid.create_spherical_grid(self.width).to(device)
        self.uvgrid = S360.grid.create_image_grid(self.width, self.height).to(device)
        self.device = device
        self.photo_params = L.photometric.PhotometricLossParameters(
            alpha=0.85, l1_estimator='none', ssim_estimator='none',
            ssim_mode='gaussian', std=1.5, window=7
        )
        self.batch_size = setting.batch_size
        self.d2n_nei = setting.d2n_nei
        self.planar_thresh = setting.planar_thresh
        self.lambda_norm_reg = setting.lambda_norm_reg
        self.lambda_planar_reg = setting.lambda_planar_reg
        self.using_normloss = setting.using_normloss
        self.using_disp2seg = setting.using_disp2seg
        

    
    def save_data(self, filename, tensor, scale=1.0):
        b, _, __, ___ = tensor.size()
        for n in range(b):
            array = tensor[n, :, :, :].detach().cpu().numpy()
            array = array.transpose(1, 2, 0) * scale
            array = numpy.float32(array)
            cv2.imwrite(filename.replace("#", str(n)), array)
    
    def forward(self, left_rgb, left_depth_pred, up_depth, up_rgb, vps, epoch, mask=None):
        
        disp = torch.cat(
            (
                torch.zeros_like(left_depth_pred),
                S360.derivatives.dtheta_vertical(self.sgrid, left_depth_pred, 0.26)  # args.baseline=0.26
            ),
            dim=1
        )
        up_render_coords = self.uvgrid + disp
        up_render_coords[torch.isnan(up_render_coords)] = 0.0
        up_render_coords[torch.isinf(up_render_coords)] = 0.0
        up_rgb_t, up_mask_t = L.splatting.render(left_rgb, left_depth_pred, \
                                                 up_render_coords, max_depth=self.depth_thresh)
    

            
        ''' Loss UD '''
        up_cutoff_mask = (up_depth < self.depth_thresh)
        up_mask_t &= ~(up_depth > self.depth_thresh)
        attention_weights = S360.weights.theta_confidence(
            S360.grid.create_spherical_grid(self.width)).to(self.device)
        # attention_weights = torch.ones_like(left_depth)
        photo1_loss = L.photometric.calculate_loss(up_rgb_t, up_rgb, self.photo_params,
                                                   mask=up_cutoff_mask, weights=attention_weights)
        active_loss = torch.tensor(0.0).to(self.device)
        active_loss += photo1_loss * 1.0  #args.photo=1.0

        ''' Loss Prior (3D Smoothness) '''
        left_xyz = S360.cartesian.coords_3d(self.sgrid, left_depth_pred)
        dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)
        guidance_duv = S360.derivatives.dI_duv(left_rgb)
        # attention_weights = torch.zeros_like(left_depth)
        depth1_smooth_loss = L.smoothness.guided_smoothness_loss(
            dI_dxyz, guidance_duv, up_cutoff_mask, (1.0 - attention_weights)
                                                   * up_cutoff_mask.type(attention_weights.dtype)
        )
        
        active_loss += depth1_smooth_loss * 0.1  #args.smooth_reg_w=0.1
        #下面部分加上两个loss
        '''Mahattan Align'''
        if self.using_normloss and self.training:  # 测refinenet时，先设置为False
            xyz = left_depth_pred.permute(0, 2, 3, 1) * get_uni_sphere_xyz(self.batch_size, self.height, self.width).to(
                self.device)  
            pred_norm = depth2norm(xyz, self.height, self.width, self.d2n_nei)  # output is b*3*h*w
            #vps = batch['vps'].to(self.device)
            mmap, mmap_mask, mmap_mask_thresh = compute_mmap(self.batch_size, pred_norm, vps, self.height, self.width, epoch,
                                                             self.d2n_nei)  
            aligned_norm = align_smooth_norm(self.batch_size, mmap, vps, self.height, self.width)
            #normal visualization
            # self.save_data("./framework_image_bak/" + str(self.count) + "_pred_norm.exr", pred_norm, scale=1.0)
            # self.save_data("./framework_image_bak/" + str(self.count) + "_aligned_norm.exr", aligned_norm, scale=1.0)
            # self.count += 1;
            '''Co-Planar'''
            if self.using_disp2seg:
                xyz = xyz.permute(0, 3, 1, 2).reshape(self.batch_size, 3, -1).float()
                # plane_loss, out_planar = generate_planar_depth(left_rgb, aligned_norm, xyz, self.width, self.height, self.device, self.batch_size,
                                                   # self.planar_thresh)  # 这里的xyz要不permute
                out_planar = generate_planar_depth(left_rgb, aligned_norm, xyz, self.width, self.height, self.device, self.batch_size,
                                                   self.planar_thresh)  # 这里的xyz要不permute
                                                   

        ''' Loss Align-normal'''
        if self.using_normloss and self.training:  # 测refinenet时，先设置为False
            loss_norm_reg = 0.0
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            norm_loss_score = cos(pred_norm, aligned_norm)  
            

            if self.using_disp2seg:
                planar_mask = out_planar['planar_mask']
                normloss_mask = mmap_mask * planar_mask
                if torch.any(torch.isnan(norm_loss_score)):
                    print('warning! nan is norm loss compute! set nan = 1')
                    norm_loss_score = torch.where(torch.isnan(norm_loss_score), torch.full_like(norm_loss_score, 1),
                                                  norm_loss_score)
            else:
                normloss_mask = mmap_mask  
            
            norm_loss = (1 - norm_loss_score).unsqueeze(1) * normloss_mask
            # norm_loss *=  attention_weights
            loss_norm_reg += torch.mean(norm_loss)
            active_loss += self.lambda_norm_reg * loss_norm_reg
            ''' Loss Planar'''
            if self.using_disp2seg :
                loss_planar_reg = 0.0
                planar_depth = out_planar['planar_depth']
                #self.saver.save_samples(up_rgb, up_depth, planar_depth, mask)
                planar_mask = out_planar['planar_mask']
                pred_depth = left_depth_pred
                
                assert torch.isnan(pred_depth).sum() == 0, print(pred_depth)

                if torch.any(torch.isnan(planar_depth)):
                    print('warning! nan in planar_depth!')
                    planar_depth = torch.where(torch.isnan(planar_depth), torch.full_like(planar_depth, 0),
                                               planar_depth)
                    pred_depth = torch.where(torch.isnan(planar_depth), torch.full_like(pred_depth, 0), pred_depth)
                planar_loss = torch.abs(pred_depth - planar_depth) * planar_mask
                # planar_loss *= attention_weights
                loss_planar_reg += torch.mean(planar_loss) 
            #loss_planar_reg = 0.0
            #loss_planar_reg = plane_loss.item()
            
            active_loss += self.lambda_planar_reg * loss_planar_reg

        return active_loss