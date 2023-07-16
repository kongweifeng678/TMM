# #读深度图
# def read_dpt(dpt_file_path):
    # """read depth map from *.dpt file.
    # :param dpt_file_path: the dpt file path
    # :type dpt_file_path: str
    # :return: depth map data
    # :rtype: numpy
    # """
    # TAG_FLOAT = 202021.25  # check for this when READING the file

    # ext = os.path.splitext(dpt_file_path)[1]

    # assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    # assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    # fid = None
    # try:
        # fid = open(dpt_file_path, 'rb')
    # except IOError:
        # print('readFlowFile: could not open %s', dpt_file_path)

    # tag = unpack('f', fid.read(4))[0]
    # width = unpack('i', fid.read(4))[0]
    # height = unpack('i', fid.read(4))[0]

    # assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    # assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    # assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # # arrange into matrix form
    # depth_data = np.fromfile(fid, np.float32)
    # depth_data = depth_data.reshape(height, width)

    # fid.close()

    # return depth_data
# #读RGB图  MP3D-2K
# def image_read(image_file_path):
    # """[summary]
    # :param image_file_path: the absolute path of image
    # :type image_file_path: str
    # :return: the numpy array of image
    # :rtype: numpy
    # """    
    # if not os.path.exists(image_file_path):
        # log.error("{} do not exist.".format(image_file_path))

    # return np.asarray(Image.open(image_file_path))
# #处理txt文件路径 data_fns是文件路径(xx/xx/xx.txt)
    # with open(opt.data_fns, 'r') as f:
        # data_fns = f.readlines()

    # if opt.sample_size > 0:
        # np.random.seed(1337)
        # data_fns = np.random.choice(data_fns, size=opt.sample_size, replace=False)


# for idx, line in enumerate(data_fns):

            # line = line.splitlines()[0].split(" ")
            # erp_image_filename = line[0]
            # erp_gtdepth_filename = line[1] if line[1] != 'None' else ""

            # if "matterport" in erp_image_filename:
                # opt.dataset_matterport_hexagon_mask_enable = True
                # opt.dataset_matterport_blur_area_height = 140  # * 0.75
            # else:
                # opt.dataset_matterport_hexagon_mask_enable = False

            # erp_pred_filename = erp_gtdepth_filename.replace("depth.dpt", "dispmap_aligned.pfm")
            # data_root = os.path.dirname(erp_image_filename)
            # debug_output_dir = os.path.join(data_root, "debug/")
            # Path(debug_output_dir).mkdir(parents=True, exist_ok=True)

            # erp_aligned_dispmap_filepath = erp_pred_filename
            # erp_image_filepath = erp_image_filename
            # erp_gt_filepath = erp_gtdepth_filename
            # filename_base, _ = os.path.splitext(os.path.basename(erp_image_filename))

            # fnc = FileNameConvention()
            # fnc.set_filename_basename(filename_base)
            # fnc.set_filepath_folder(debug_output_dir)

            # # load ERP rgb image and estimate the ERP depth map
            # erp_rgb_image_data = image_io.image_read(erp_image_filepath)
            # # Load matrices for blending linear system
            # estimated_depthmap, times = depthmap_estimation(erp_rgb_image_data, fnc, opt, blend_it, iter)

            # # get error fo ERP depth map
            # erp_gt_depthmap = depthmap_utils.read_dpt(erp_gt_filepath) if erp_gt_filepath != "" else None
            # pred_metrics = error_metric(estimated_depthmap, erp_gt_depthmap) if erp_gt_filepath != "" else None

            # serialization.save_predictions(output_folder, erp_gt_depthmap, erp_rgb_image_data, estimated_depthmap,
                                           # opt.persp_monodepth, idx=idx)

from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms
import os
from struct import unpack
import numpy as np
from struct import unpack
import os
import sys
import re
import gc
from .util import Equirec2Cube
import torch
import cv2
class MP3D_2K(Dataset):
    def __init__(self, data_fns, delimiter = " "):
        #fileID = open(filenamesFilePath, "r")
        fileID = open(data_fns, "r")
        self.lines = fileID.readlines()
        self.pilToTensor =  transforms.ToTensor()
        self.delimiter = delimiter
        self.sample = {} 
        self.sample["leftRGB"] = []
        self.sample["leftDepth"] = []
        self.max_depth_meters = 10.0
        # self.height = 1024         
        # self.width = 2048
        self.height = 256         
        self.width = 512
        self.e2c = Equirec2Cube(self.height, self.width, self.height // 2)
        for line in self.lines:
            leftRGBPath = line.split(self.delimiter)[0]
            leftDepthPath = line.split(self.delimiter)[1]
            if '\n' in leftDepthPath:
                leftDepthPath = leftDepthPath[:-1]
            self.sample["leftRGB"].append(leftRGBPath)
            self.sample["leftDepth"].append(leftDepthPath)
            
    def __getitem__(self, idx):
        #leftRGB = Image.open(self.sample["leftRGB"][idx])
        leftRGB = cv2.imread(self.sample["leftRGB"][idx])
        leftRGB = cv2.resize(leftRGB, (self.width,self.height))
        leftRGB = cv2.cvtColor(leftRGB, cv2.COLOR_BGR2RGB)
        cube_leftRGB = self.e2c.run(leftRGB)
        leftDepth = self.read_dpt(self.sample["leftDepth"][idx])
        leftDepth = cv2.resize(leftDepth, (self.width,self.height))
        leftDepth = torch.from_numpy(leftDepth)
        leftDepth.unsqueeze_(0)
        
        inputs = {
            "rgb" : self.pilToTensor(leftRGB),
            "leftDepth": leftDepth,
            "cube_rgb": self.pilToTensor(cube_leftRGB)
        }
        # inputs["val_mask"] = ((inputs["upDepth"] > 0) & (inputs["upDepth"] <= self.max_depth_meters)
                                  # & ~torch.isnan(inputs["upDepth"]))
        # print(intputs["rgb"].size())
        # print(intputs["leftDepth"].size())
        # exit()
        return inputs
        
    def __len__(self):
        return len(self.lines)
        
    def read_dpt(self, dpt_file_path):
        """read depth map from *.dpt file.
        :param dpt_file_path: the dpt file path
        :type dpt_file_path: str
        :return: depth map data
        :rtype: numpy
        """
        TAG_FLOAT = 202021.25  # check for this when READING the file

        ext = os.path.splitext(dpt_file_path)[1]
        #ext = dpt_file_path

        assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
        assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)
        
        fid = None
        try:
            fid = open(dpt_file_path, 'rb')
        except IOError:
            print('readFlowFile: could not open %s', dpt_file_path)

        tag = unpack('f', fid.read(4))[0]
        width = unpack('i', fid.read(4))[0]
        height = unpack('i', fid.read(4))[0]

        assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
        assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
        assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

        # arrange into matrix form
        depth_data = np.fromfile(fid, np.float32)
        depth_data = depth_data.reshape(height, width)

        fid.close()

        return depth_data