import matplotlib.pyplot as plt
import cv2
import numpy as np
for i in range(0,11,1):
    depth_pred = cv2.imread('%d_aligned_norm.exr'%(i), cv2.IMREAD_UNCHANGED)
    invalid_mask = depth_pred > 254
    depth_pred = (depth_pred + 1.0)/ 2.0
    depth_pred[invalid_mask] = 0
    #depth_pred = depth_pred.astype(np.float32)/20
    fig, ax = plt.subplots()
    #ax.imshow(depth_pred, plt.cm.plasma)
    ax.imshow(depth_pred)
    plt.axis('off')
    height, width, lend= depth_pred.shape
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig('./aligned_norm/%d_aligned_norm.png'%(i), dpi=300)
    plt.close()