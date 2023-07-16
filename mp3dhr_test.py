from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import numpy
import torch
from torch.utils.data import DataLoader

from networks import UniFuse, Equi, UniFuse_weight
import datasets
from metrics import Evaluator
from saver import Saver

import supervision as L
#import exporters as IO
import spherical as S360
from saver import Saver

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")

parser.add_argument("--data_path", default="/home/wolian/disk1/Stanford2D3D/", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="matterport3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d", "my_3d60", "MP3DHR"],
                    type=str, help="dataset to evaluate on.")

parser.add_argument("--load_weights_dir", type=str, help="folder of model to load")

parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=6, help="batch size")

parser.add_argument("--median_align", action="store_true", help="if set, apply median alignment in evaluation")
parser.add_argument("--save_samples", action="store_true", help="if set, save the depth maps and point clouds")
parser.add_argument("--depth_thresh", type=float, default=10.0, help = "Depth threshold - depth clipping.")
parser.add_argument("--median_scale", required=False, default=False, action="store_true", help = "Perform median scaling before calculating metrics.")
parser.add_argument("--spherical_weights", required=False, default=False, action="store_true", help = "Use spherical weighting when calculating the metrics.")
parser.add_argument("--spherical_sampling", required=False, default=False, action="store_true", help = "Use spherical sampling when calculating the metrics.")

settings = parser.parse_args()

def compute_errors(gt, pred, invalid_mask, weights, sampling, mode='cpu', median_scale=False):
    b, _, __, ___ = gt.size()
    scale = torch.median(gt.reshape(b, -1), dim=1)[0] / torch.median(pred.reshape(b, -1), dim=1)[0]\
        if median_scale else torch.tensor(1.0).expand(b, 1, 1, 1).to(gt.device) 
    pred = pred * scale.reshape(b, 1, 1, 1)
    valid_sum = torch.sum(~invalid_mask, dim=[1, 2, 3], keepdim=True)
    gt[invalid_mask] = 0.0
    pred[invalid_mask] = 0.0
    
    gt[gt<0.1] = 0.1
    pred[pred<0.1] = 0.1
    gt[gt>10] = 10
    gt[pred>10] = 10
    
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

    # data
    datasets_dict = {"3d60": datasets.ThreeD60,
                     "panosuncg": datasets.PanoSunCG,
                     "stanford2d3d": datasets.Stanford2D3D,
                     "matterport3d": datasets.Matterport3D,
                     "my_3d60": datasets.Dataset360D,
                     "MP3DHR": datasets.MP3D_2K}
    dataset = datasets_dict[settings.dataset]

    fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

    test_file_list = fpath.format(settings.dataset, "test")
    #test_dataset = dataset(settings.data_path, " ", "ud", [256, 512], "/data/kongweifeng/GithubFile/360Depth/ss360/SVS_norm/vps/") #3d60
    test_dataset = dataset(settings.data_path)
    # test_dataset = dataset(settings.data_path, test_file_list,
                           # model_dict['height'], model_dict['width'], is_training=False)
    test_loader = DataLoader(test_dataset, settings.batch_size, False,
                             num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // settings.batch_size
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    Net_dict = {"UniFuse": UniFuse,
                "Equi": Equi,
                "UniFuse_weight": UniFuse_weight}
    Net = Net_dict[model_dict['net']]

    model = Net(model_dict['layers'], model_dict['height'], model_dict['width'],
                max_depth=test_dataset.max_depth_meters, fusion_type=model_dict['fusion'],
                se_in_fusion=model_dict['se_in_fusion'])

    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    # evaluator = Evaluator(settings.median_align)
    # evaluator.reset_eval_metrics()
    #saver = Saver(load_weights_folder)
    saver = Saver("./vis/MP3D-2K/")
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")
    #width = 512
    #height = 256
    width = 512
    height = 256
    #errors = numpy.zeros((7, 1484), numpy.float32)  #mp3d_teset 1484   sf3d_test 346  mp3dhr 1850
    errors = numpy.zeros((7, 1850), numpy.float32)
    weights = S360.weights.theta_confidence(
        S360.grid.create_spherical_grid(width)
    ).to(device) if settings.spherical_weights else torch.ones(1, 1, height, width).to(device)
    sampling = spiral_sampling(S360.grid.create_image_grid(width, height), 0.25).to(device) \
        if settings.spherical_sampling else torch.ones(1, 1, height, width).to(device)
    with torch.no_grad():
        for test_batch_id, inputs in enumerate(pbar):
            #equi_inputs = inputs["normalized_rgb"].to(device)
            equi_inputs = inputs["rgb"].to(device)
            #cube_inputs = inputs["normalized_cube_rgb"].to(device)
            cube_inputs = inputs["cube_rgb"].to(device)
            outputs = model(equi_inputs, cube_inputs)
            
            #pred_depth = outputs["pred_depth"].detach().cpu()
            left_depth_pred = outputs["pred_depth"]
            left_depth_pred = torch.abs(left_depth_pred)
            left_depth_pred[left_depth_pred < 0.1] = 0.1
            #left_depth_pred[left_depth_pred < 1] = 1
            left_depth_pred[left_depth_pred > 10.0] = 10.0
            
            
            #left_depth_pred[torch.isnan(left_depth_pred)] = 0.1 
            #gt_depth = inputs["gt_depth"]
            left_depth = inputs['leftDepth'].to(device)
            

            #mask = (left_depth > settings.depth_thresh | torch.isnan(left_depth) | left_depth < 0) 
            #mask = ((left_depth > settings.depth_thresh) | (torch.isnan(left_depth)) | (left_depth < 0.01))
            mask = ((left_depth > settings.depth_thresh) | (torch.isnan(left_depth)) | (left_depth < 0.1))
            b, c, h, w = equi_inputs.size() 
            #mask = inputs["val_mask"]
            # for i in range(gt_depth.shape[0]):
                # evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])
            ''' Errors '''
            abs_rel_t, sq_rel_t, rmse_t, rmse_log_t, delta1, delta2, delta3\
                = compute_errors(left_depth, left_depth_pred, mask, weights=weights, sampling=sampling, \
                    mode='gpu', \
                    median_scale=settings.median_scale)
                 
            ''' Visualize & Append Errors '''
            for i in range(b):
                #idx = counter + i
                idx = test_batch_id * settings.batch_size + i
                errors[:, idx] = abs_rel_t[0][i], sq_rel_t[0][i], rmse_t[0][i], \
                    rmse_log_t[0][i], delta1[i], delta2[i], delta3[i]
            if settings.save_samples:
                saver.save_samples(inputs["rgb"], left_depth, left_depth_pred, mask)

            
        mean_errors = errors.mean(1)
        error_names = ['abs_rel','sq_rel','rmse','log_rmse','delta1','delta2','delta3']
        #print("Results ({}): ".format(settings.name))
        print("\t{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("\t{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors))        
 


    #evaluator.print(load_weights_folder)


if __name__ == "__main__":
    main()