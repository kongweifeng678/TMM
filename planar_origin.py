import torch
import numpy as np
from layers import *
from skimage.segmentation import all_felzenszwalb as felz_seg
import torch.nn.functional as F
from sphere_xyz import get_uni_sphere_xyz
import matplotlib.pyplot as plt

def generate_planar_depth(rgb, aligned_norm, xyz, width, height, device, batch_size=16, planar_thresh=200):   # args width height batch_size planar_thresh
    cam_points = xyz  # 注意原来的第四个索引有什么作用
    segment = compute_seg(xyz, aligned_norm, rgb, width, height, batch_size).unsqueeze(1)  # 这里的输入数据还没填
    '''visualizaton'''
    
    max_num = segment.max().item() + 1
    sum_points = torch.zeros((batch_size, max_num, 3)).to(device)
    area = torch.zeros((batch_size, max_num)).to(device)
    for channel in range(3):
        points_channel = sum_points[:, :, channel]
        points_channel = points_channel.reshape(batch_size, -1)
        points_channel.scatter_add_(1, segment.view(batch_size, -1),
                                    cam_points[:, channel, ...].view(batch_size, -1))

    area.scatter_add_(1, segment.view(batch_size, -1), torch.ones_like(rgb).view(batch_size, -1))
    # X^T X
    cam_points_tmp = cam_points
    x_T_dot_x = (cam_points_tmp.unsqueeze(1) * cam_points_tmp.unsqueeze(2))
    x_T_dot_x = x_T_dot_x.view(batch_size, 9, -1)
    X_T_dot_X = torch.zeros((batch_size, max_num, 9)).cuda()
    for channel in range(9):
        points_channel = X_T_dot_X[:, :, channel]
        points_channel = points_channel.reshape(batch_size, -1)
        points_channel.scatter_add_(1, segment.view(batch_size, -1),
                                    x_T_dot_x[:, channel, ...].view(batch_size, -1))
    xTx = X_T_dot_X.view(batch_size, max_num, 3, 3)

    # take inverse
    xTx_inv = mat_3x3_inv(
        xTx.view(-1, 3, 3) + 0.01 * torch.eye(3).view(1, 3, 3).expand(batch_size * max_num, 3, 3).cuda())
    xTx_inv = xTx_inv.view(xTx.shape)
    xTx_inv_xT = torch.matmul(xTx_inv, sum_points.unsqueeze(3))
    plane_parameters = xTx_inv_xT.squeeze(3)

    # generate mask for segment with area larger than 200
    planar_area_thresh = planar_thresh
    valid_mask = (area > planar_area_thresh).float()
    planar_mask = torch.gather(valid_mask, 1, segment.view(batch_size, -1))
    planar_mask = planar_mask.view(batch_size, 1, height, width)
    planar_mask[:, :, :8, :] = 0
    planar_mask[:, :, -8:, :] = 0
    planar_mask[:, :, :, :8] = 0
    planar_mask[:, :, :, -8:] = 0
    outputs = {}
    outputs["planar_mask"] = planar_mask
    # segment unpooling
    unpooled_parameters = []
    for channel in range(3):
        pooled_parameters_channel = plane_parameters[:, :, channel]
        pooled_parameters_channel = pooled_parameters_channel.reshape(batch_size, -1)
        unpooled_parameter = torch.gather(pooled_parameters_channel, 1, segment.view(batch_size, -1))
        unpooled_parameters.append(unpooled_parameter.view(batch_size, 1, height, width))
    unpooled_parameters = torch.cat(unpooled_parameters, dim=1)
    K_inv_dot_xy1 = get_uni_sphere_xyz(batch_size, height, width).permute(0, 3, 1, 2).reshape(batch_size, 3, -1).cuda()
    depth = 1. / (torch.sum(K_inv_dot_xy1 * unpooled_parameters.view(batch_size, 3, -1), dim=1) + 1e-6)

    # clip depth range
    depth = torch.clamp(depth, 0.0, 10.0)  # 这里的clamp范围可能还要确定
    depth = depth.view(batch_size, 1, height, width)
    outputs["planar_depth"] = depth
    outputs["segment"] = segment
    return outputs

def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)


def mat_3x3_det(mat):
    '''
    calculate the determinant of a 3x3 matrix, support batch.
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    det = mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) \
        - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]) \
        + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
    return det


#  generate planar depth
def mat_3x3_inv(mat):
    '''
    calculate the inverse of a 3x3 matrix, support batch.
    :param mat: torch.Tensor -- [input matrix, shape: (B, 3, 3)]
    :return: mat_inv: torch.Tensor -- [inversed matrix shape: (B, 3, 3)]
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    # Divide the matrix with it's maximum element
    max_vals = mat.max(1)[0].max(1)[0].view((-1, 1, 1))
    mat = mat / max_vals

    det = mat_3x3_det(mat)
    inv_det = 1.0 / det

    mat_inv = torch.zeros(mat.shape, device=mat.device)
    mat_inv[:, 0, 0] = (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 0, 1] = (mat[:, 0, 2] * mat[:, 2, 1] - mat[:, 0, 1] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 0, 2] = (mat[:, 0, 1] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 1, 0] = (mat[:, 1, 2] * mat[:, 2, 0] - mat[:, 1, 0] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 1, 1] = (mat[:, 0, 0] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 0]) * inv_det
    mat_inv[:, 1, 2] = (mat[:, 1, 0] * mat[:, 0, 2] - mat[:, 0, 0] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 2, 0] = (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 2, 0] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 2, 1] = (mat[:, 2, 0] * mat[:, 0, 1] - mat[:, 0, 0] * mat[:, 2, 1]) * inv_det
    mat_inv[:, 2, 2] = (mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 1, 0] * mat[:, 0, 1]) * inv_det

    # Divide the maximum value once more
    mat_inv = mat_inv / max_vals
    return mat_inv


def compute_seg(xyz, aligned_norm, rgb, width, height, batch_size, nei=3):
    """
    inputs:
        cam_points             b, 4, H*W
        aligned_norm           b, 3, H, W
        rgb                    b, 3, H, W
    outputs:
        seg                b, 1, H, W
    """
    cam_points = xyz
    # calculate D using aligned norm
    D = compute_D(cam_points, aligned_norm)  # plane-to-origin
    D = D.reshape(batch_size, height, width)
    # move valid border from depth2norm neighborhood
    rgb = rgb[:, :, 2 * nei:, :-2 * nei]
    D = D[:, 2 * nei:, :-2 * nei]
    aligned_norm = aligned_norm[:, :, 2 * nei:, :-2 * nei]
    # comute cost
    # RGB D norm均计算与上下左右点的欧氏距离
    pdist = torch.nn.PairwiseDistance(p=2)
    rgb_down = pdist(rgb[:, :, 1:], rgb[:, :, :-1])
    rgb_right = pdist(rgb[:, :, :, 1:], rgb[:, :, :, :-1])
    rgb_down = torch.stack([normalize(rgb_down[i]) for i in range(batch_size)])
    rgb_right = torch.stack([normalize(rgb_right[i]) for i in range(batch_size)])

    D_down = abs(D[:, 1:] - D[:, :-1])
    D_right = abs(D[:, :, 1:] - D[:, :, :-1])
    norm_down = pdist(aligned_norm[:, :, 1:], aligned_norm[:, :, :-1])
    norm_right = pdist(aligned_norm[:, :, :, 1:], aligned_norm[:, :, :, :-1])

    D_down = torch.stack([normalize(D_down[i]) for i in range(batch_size)])
    norm_down = torch.stack([normalize(norm_down[i]) for i in range(batch_size)])

    D_right = torch.stack([normalize(D_right[i]) for i in range(batch_size)])
    norm_right = torch.stack([normalize(norm_right[i]) for i in range(batch_size)])

    # geometric dissimilarity
    normD_down = D_down + norm_down
    normD_right = D_right + norm_right

    normD_down = torch.stack([normalize(normD_down[i]) for i in range(batch_size)])
    normD_right = torch.stack([normalize(normD_right[i]) for i in range(batch_size)])

    # get max from (rgb, normD)
    cost_down = torch.stack([rgb_down, normD_down])
    cost_right = torch.stack([rgb_right, normD_right])
    cost_down, _ = torch.max(cost_down, 0)
    cost_right, _ = torch.max(cost_right, 0)

    # felz_seg 这里的分割要看一下
    cost_down_np = cost_down.detach().cpu().numpy()
    cost_right_np = cost_right.detach().cpu().numpy()
    segment = torch.stack([torch.from_numpy(
        felz_seg(normalize(cost_down_np[i]), normalize(cost_right_np[i]), 0, 0, height - 2 * nei,
                 width - 2 * nei, scale=1, min_size=50)).cuda() for i in range(batch_size)])
    # pad the edges that were previously trimmed
    segment += 1
    segment = F.pad(segment, (0, 2 * nei, 2 * nei, 0), "constant", 0)
    return segment
