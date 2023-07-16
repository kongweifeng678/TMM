from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import numpy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from networks import UniFuse, Equi, UniFuse_weight, UniFuse_weight_onebranch, UniFuse_weight_sphericalpad
import datasets
from metrics import Evaluator
#from saver import Saver
import matplotlib.pyplot as plt
import numpy as np
import supervision as L
#import exporters as IO
import spherical as S360
from util import Equirec2Cube
import cv2
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time
parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")

parser.add_argument("--data_path", default="/home/wolian/disk1/Stanford2D3D/", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="matterport3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d", "my_3d60"],
                    type=str, help="dataset to evaluate on.")

parser.add_argument("--load_weights_dir", type=str, help="folder of model to load")

parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")

parser.add_argument("--median_align", action="store_true", help="if set, apply median alignment in evaluation")
parser.add_argument("--save_samples", action="store_true", help="if set, save the depth maps and point clouds")
parser.add_argument("--depth_thresh", type=float, default=10.0, help = "Depth threshold - depth clipping.")
parser.add_argument("--median_scale", required=False, default=False, action="store_true", help = "Perform median scaling before calculating metrics.")
parser.add_argument("--spherical_weights", required=False, default=False, action="store_true", help = "Use spherical weighting when calculating the metrics.")
parser.add_argument("--spherical_sampling", required=False, default=False, action="store_true", help = "Use spherical sampling when calculating the metrics.")
parser.add_argument("--save_path", default="./vis/output/", type=str, help="path to the visual result.")

settings = parser.parse_args()
def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass
class Saver(object):

    def __init__(self, save_dir):
        self.idx = 0
        self.save_dir = os.path.join(save_dir, "results")
        #self.save_dir = os.path.join(save_dir, "my_test")
        
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)
 
    def save_as_point_cloud(self, depth, rgb, path, mask=None):
        h, w = depth.shape
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = -np.repeat(Phi, h, axis=0)

        X = depth * np.sin(Theta) * np.sin(Phi)
        Y = depth * np.cos(Theta)
        Z = depth * np.sin(Theta) * np.cos(Phi)

        if mask is None:
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            R = rgb[:, :, 0].flatten()
            G = rgb[:, :, 1].flatten()
            B = rgb[:, :, 2].flatten()
        else:
            X = X[mask]
            Y = Y[mask]
            Z = Z[mask]
            R = rgb[:, :, 0][mask]
            G = rgb[:, :, 1][mask]
            B = rgb[:, :, 2][mask]

        XYZ = np.stack([X, Y, Z], axis=1)
        RGB = np.stack([R, G, B], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(RGB)
        o3d.io.write_point_cloud(path, pcd)
    
    #oringin_version
    # def save_samples(self, rgbs, gt_depths, pred_depths, depth_masks=None):
        # """
        # Saves samples
        # """
        # rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        # depth_preds = pred_depths.cpu().numpy()
        # gt_depths = gt_depths.cpu().numpy()
        # depth_masks = None
        # if depth_masks is None:
            # depth_masks = gt_depths != 0
        # else:
            # depth_masks = depth_masks.cpu().numpy()

        # for i in range(rgbs.shape[0]):
            # self.idx = self.idx+1
            # mkdirs(os.path.join(self.save_dir, '%04d'%(self.idx)))

            # cmap = plt.get_cmap("rainbow_r")

            # depth_pred = cmap(depth_preds[i][0].astype(np.float32)/10)
            # depth_pred = np.delete(depth_pred, 3, 2)
            # path = os.path.join(self.save_dir, '%04d' % (self.idx) ,'_depth_pred.jpg')
            # #path = os.path.join(self.save_dir, 'test' ,'%04d.jpg' % (self.idx))
            # cv2.imwrite(path, (depth_pred * 255).astype(np.uint8))

            # depth_gt = cmap(gt_depths[i][0].astype(np.float32)/10)
            # depth_gt = np.delete(depth_gt, 3, 2)
            # # depth_gt[..., 0][~depth_masks[i][0]] = 0
            # # depth_gt[..., 1][~depth_masks[i][0]] = 0
            # # depth_gt[..., 2][~depth_masks[i][0]] = 0
            # path = os.path.join(self.save_dir, '%04d' % (self.idx), '_depth_gt.jpg')
            # cv2.imwrite(path, (depth_gt * 255).astype(np.uint8))

            # # path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_pc_pred.ply')
            # # self.save_as_point_cloud(depth_preds[i][0], rgbs[i], path)

            # # path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_pc_gt.ply')
            # # self.save_as_point_cloud(gt_depths[i][0], rgbs[i], path, depth_masks[i][0])

            # rgb = (rgbs[i] * 255).astype(np.uint8)
            # path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_rgb.jpg')
            # cv2.imwrite(path, rgb[:,:,::-1])
    
    
    #kaitidabian_version
    def save_samples(self, pred_depths, img_name, depth_masks=None):
        """
        Saves samples
        """
        #rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        depth_preds = pred_depths.cpu().numpy()
        
        #print(depth_preds)
        
        #gt_depths = gt_depths.cpu().numpy()
        
        #print(gt_depths)
        

        for i in range(1):
            self.idx = self.idx+1
            #wilson
            #mkdirs(os.path.join(self.save_dir, '%04d'%(self.idx)))

            cmap = plt.get_cmap("rainbow_r")

            depth_pred = cmap(depth_preds[i][0].astype(np.float32)/10)
            depth_pred = np.delete(depth_pred, 3, 2)
            
            #wilson
            #path = os.path.join(self.save_dir, 'depth_pred_%04d.jpg' % (self.idx))
            img_name = 'depth_pred_' + img_name
            path = os.path.join(self.save_dir, img_name)
            img_name
            #path = os.path.join(self.save_dir, 'test' ,'%04d.jpg' % (self.idx))
            
            print(path)
            
            #wilson
            cv2.imwrite(path, (depth_pred * 255).astype(np.uint8))
            
            #wilson
            # path = os.path.join(self.save_dir, 'gt_pred_%04d.jpg' % (self.idx))
            # cv2.imwrite(path, (depth_pred * 255).astype(np.uint8))

            # depth_gt = cmap(gt_depths[i][0].astype(np.float32)/10)
            # depth_gt = np.delete(depth_gt, 3, 2)
            # depth_gt[..., 0][~depth_masks[i][0]] = 0
            # depth_gt[..., 1][~depth_masks[i][0]] = 0
            # depth_gt[..., 2][~depth_masks[i][0]] = 0
            # path = os.path.join(self.save_dir, '%04d' % (self.idx), '_depth_gt.jpg')
            # cv2.imwrite(path, (depth_gt * 255).astype(np.uint8))

            # path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_pc_pred.ply')
            # self.save_as_point_cloud(depth_preds[i][0], rgbs[i], path)

            # path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_pc_gt.ply')
            # self.save_as_point_cloud(gt_depths[i][0], rgbs[i], path, depth_masks[i][0])
    
            #wilson
            # rgb = (rgbs[i] * 255).astype(np.uint8)
            # path = os.path.join(self.save_dir, 'rgb_%04d.jpg'%(self.idx))
            # cv2.imwrite(path, rgb[:,:,::-1])



def compute_errors(gt, pred, invalid_mask, weights, sampling, mode='cpu', median_scale=False):
    b, _, __, ___ = gt.size()
    scale = torch.median(gt.reshape(b, -1), dim=1)[0] / torch.median(pred.reshape(b, -1), dim=1)[0]\
        if median_scale else torch.tensor(1.0).expand(b, 1, 1, 1).to(gt.device) 
    pred = pred * scale.reshape(b, 1, 1, 1)
    valid_sum = torch.sum(~invalid_mask, dim=[1, 2, 3], keepdim=True)
    
    gt[gt < 0.1] = 0.1
    gt[gt > 10.0] = 10.0
    pred[pred < 0.1] = 0.1
    pred[pred > 10.0] = 10.0
    
    gt[invalid_mask] = 0.0
    pred[invalid_mask] = 0.0

    thresh = torch.max((gt / pred), (pred / gt))
    thresh[invalid_mask | (sampling < 0.5)] = 2.0
    
    sum_dims = [1, 2, 3]
    delta_valid_sum = torch.sum(~invalid_mask & (sampling > 0), dim=[1, 2, 3], keepdim=True)
    delta1 = (thresh < 1.25   ).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
    delta2 = (thresh < (1.25 ** 2)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
    delta3 = (thresh < (1.25 ** 3)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()

    rmse = (gt - pred) ** 2
    rmse[invalid_mask] = 0.0
    rmse_w = rmse * weights
    rmse_mean = torch.sqrt(rmse_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log[invalid_mask] = 0.0
    rmse_log_w = rmse_log * weights
    rmse_log_mean = torch.sqrt(rmse_log_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

    abs_rel = (torch.abs(gt - pred) / gt)
    abs_rel[invalid_mask] = 0.0
    abs_rel_w = abs_rel * weights
    abs_rel_mean = abs_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

    sq_rel = (((gt - pred)**2) / gt)
    sq_rel[invalid_mask] = 0.0
    sq_rel_w = sq_rel * weights
    sq_rel_mean = sq_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

    return (abs_rel_mean, abs_rel), (sq_rel_mean, sq_rel), (rmse_mean, rmse), \
        (rmse_log_mean, rmse_log), delta1, delta2, delta3

def spiral_sampling(grid, percentage):
    b, c, h, w = grid.size()    
    N = torch.tensor(h*w*percentage).int().float()    
    sampling = torch.zeros_like(grid)[:, 0, :, :].unsqueeze(1)
    phi_k = torch.tensor(0.0).float()
    for k in torch.arange(N - 1):
        k = k.float() + 1.0
        h_k = -1 + 2 * (k - 1) / (N - 1)
        theta_k = torch.acos(h_k)
        phi_k = phi_k + torch.tensor(3.6).float() / torch.sqrt(N) / torch.sqrt(1 - h_k * h_k) \
            if k > 1.0 else torch.tensor(0.0).float()
        phi_k = torch.fmod(phi_k, 2 * numpy.pi)
        sampling[:, :, int(theta_k / numpy.pi * h) - 1, int(phi_k / numpy.pi / 2 * w) - 1] += 1.0
    return (sampling > 0).float()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_weights_folder = os.path.expanduser(settings.load_weights_dir)
    model_path = os.path.join(load_weights_folder, "model.pth")
    model_dict = torch.load(model_path)

    # # data
    # datasets_dict = {"3d60": datasets.ThreeD60,
                     # "panosuncg": datasets.PanoSunCG,
                     # "stanford2d3d": datasets.Stanford2D3D,
                     # "matterport3d": datasets.Matterport3D,
                     # "my_3d60": datasets.Dataset360D}
    # dataset = datasets_dict[settings.dataset]

    # fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

    # test_file_list = fpath.format(settings.dataset, "test")
    
    # #3d60
    # test_dataset = dataset(settings.data_path, " ", "ud", [256, 512], "/data/kongweifeng/GithubFile/360Depth/ss360/SVS_norm/vps/")

    # test_dataset = dataset(settings.data_path, test_file_list,
                           # model_dict['height'], model_dict['width'], is_training=False)
    #shanghai360
    # test_dataset = dataset("", settings.data_path,
                           # model_dict['height'], model_dict['width'], is_training=False)
                           
    # test_loader = DataLoader(test_dataset, settings.batch_size, False,
                             # num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    # num_test_samples = len(test_dataset)
    # num_steps = num_test_samples // settings.batch_size
    #wilson
    # print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    Net_dict = {"UniFuse": UniFuse,
                "Equi": Equi,
                "UniFuse_weight": UniFuse_weight,
                "UniFuse_weight_onebranch": UniFuse_weight_onebranch,
                "UniFuse_weight_sphericalpad": UniFuse_weight_sphericalpad}
    Net = Net_dict[model_dict['net']]
    print(model_dict['net'])
    model = Net(model_dict['layers'], model_dict['height'], model_dict['width'],
                max_depth=10.0, fusion_type=model_dict['fusion'],
                se_in_fusion=model_dict['se_in_fusion'])
    
    #calculate params
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    #end calculate



    model.to(device)
    
    #calculate FLops
    tensor_erp =  (torch.rand(1, 3, 256, 512).cuda(),)
    #tensor_erp =  (torch.rand(1, 3, 256, 512),)
    #tensor_cmp = (torch.rand(1, 3, 128, 768),)
    print(tensor_erp)
    print(type(tensor_erp))
    flops = FlopCountAnalysis(model, tensor_erp)
    print("FLOPs: ", flops.total())
    #end calculate
    
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    # evaluator = Evaluator(settings.median_align)
    # evaluator.reset_eval_metrics()
    #saver = Saver(load_weights_folder)
    saver = Saver(settings.save_path)
    
    
    width = 512
    height = 256

    with torch.no_grad():
        #equi_inputs = inputs["normalized_rgb"].to(device)
        pilToTensor =  transforms.ToTensor()
        img_path = os.listdir(settings.data_path)
        for img_name in img_path:
            img_name_final = os.path.join(settings.data_path, img_name)
            leftRGB = cv2.imread(img_name_final)
            leftRGB = cv2.resize(leftRGB, (width,height))
            leftRGB = cv2.cvtColor(leftRGB, cv2.COLOR_BGR2RGB)
            e2c = Equirec2Cube(height, width, height // 2)
            cube_leftRGB = e2c.run(leftRGB)
            
            leftRGB = pilToTensor(leftRGB)
            cube_leftRGB = pilToTensor(cube_leftRGB)
            
            leftRGB = torch.unsqueeze(leftRGB, dim=0)
            cube_leftRGB = torch.unsqueeze(cube_leftRGB, dim=0)
            
            equi_inputs = leftRGB.to(device)
            #cube_inputs = inputs["normalized_cube_rgb"].to(device)
            cube_inputs = cube_leftRGB.to(device)
            print(cube_inputs.size())
            #wilson
            # print(type(cube_rgb))
            # cube_rgb = cube_inputs.detach().cpu().numpy()
            # save_result = "./my_cmp.jpg"
            # cv2.imwrite(save_result, cube_rgb)
            # exit()
            
            start = time.time()
            outputs = model(equi_inputs, cube_inputs)
            end = time.time()
            print("time: %.5f (s)"%(end -start))
            #pred_depth = outputs["pred_depth"].detach().cpu()
            left_depth_pred = outputs["pred_depth"]
            left_depth_pred = torch.abs(left_depth_pred)
            left_depth_pred[left_depth_pred < 0.1] = 0.1
            left_depth_pred[left_depth_pred > 10.0] = 10.0
            print(left_depth_pred.size())
            print(leftRGB.size())
            saver.save_samples(left_depth_pred, img_name)
            #others
            #gt_depth = inputs["gt_depth"]
            
            
            #gt_depth[gt_depth < 0.1] = 0.1
            #gt_depth[gt_depth > 10.0] = 10.0
            

            
            
            
            # left_depth = inputs["gt_depth"].to(device)
            # left_depth[left_depth < 0.1] = 0.1
            # left_depth[left_depth > 10.0] = 10.0


                

            ''' Visualize & Append Errors '''

            # if settings.save_samples:
            
                # #wilson
                # mask=None
                # #saver.save_samples(inputs["cube_rgb"], left_depth, left_depth_pred, mask)
                # saver.save_samples(inputs["rgb"], left_depth, left_depth_pred)
                # #saver.save_samples(inputs["rgb"], left_depth, left_depth)
                # #exit(0)
        
            


        #evaluator.print(load_weights_folder)


if __name__ == "__main__":
    main()