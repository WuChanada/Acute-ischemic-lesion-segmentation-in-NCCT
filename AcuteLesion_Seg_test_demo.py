
import numpy as np

import nibabel as nib
import os
import time

from sklearn.externals import joblib

from scipy.ndimage import median_filter
from scipy.ndimage import morphology
from scipy import ndimage

fidx = [0, 1, 2, 3, 4, 5, 6, 7]
clfname1 = 'model.txt'  # model path

clf = joblib.load(clfname1)



im_root = 'test_example/'
lst = os.listdir(im_root)
imglst = lst #lst[1:]
N = len(imglst)
for ind in range(0,N):

    #
    print(imglst[ind])
    start = time.time()

# compute three scale intensity feature
    niiname = im_root + imglst[ind] + '/NCCT_brain.nii.gz'
    if os.path.isfile(niiname) == False:
        continue
    nii = nib.load(niiname)
    im = nii.get_data()
    im = im.astype(np.float32)
    hdr1 = nii.header
    aff1 = nii.affine
    roi = np.where((im > 10) & (im < 50))
    # med 3
    im3 = median_filter(im,[3, 3, 3])

    mu, sigma = np.mean(im3[roi]), np.std(im3[roi])
    eps = 1e-6
    im3 = (im3-mu) / (sigma + eps)
    im3 = (im3 - np.min(im3)) / (np.max(im3) - np.min(im3))

    # med 7
    im7 = median_filter(im,[7, 7, 7])

    mu, sigma = np.mean(im7[roi]), np.std(im7[roi])
    eps = 1e-6
    im7 = (im7-mu) / (sigma + eps)
    im7 = (im7 - np.min(im7)) / (np.max(im7) - np.min(im7))

    # med 11
    im11 = median_filter(im,[11, 11, 11])

    mu, sigma = np.mean(im11[roi]), np.std(im11[roi])
    eps = 1e-6
    im11 = (im11-mu) / (sigma + eps)
    im11 = (im11 - np.min(im11)) / (np.max(im11) - np.min(im11))

# load distance feature
    niiname = im_root + imglst[ind] + '/dist.nii.gz'
    nii = nib.load(niiname)
    dist = nii.get_data()
    dist = dist.astype(np.float32)
    # load distance feature
    niiname = im_root + imglst[ind] + '/CSF_Unet_prob.nii.gz'
    nii = nib.load(niiname)
    csfprob = nii.get_data()
    csfprob = csfprob.astype(np.float32)

    imsize = im.shape
    H = imsize[0]
    W = imsize[1]
    Z = imsize[2]

# compute three scale difference feature
    niiname = im_root + imglst[ind] + '/Diff_map.nii.gz'
    nii = nib.load(niiname)
    dif= nii.get_data()
    dif = dif.astype(np.float32)
    hdr = nii.header
    aff = nii.affine

    # med 3
    dif3 = median_filter(dif,[3, 3, 3])
    dif3[dif3>5] = 5
    dif3[dif3<-5] = -5
    dif3 = (dif3 - np.min(dif3)) / (np.max(dif3) - np.min(dif3))

    # med 7
    dif7 = median_filter(dif,[7, 7, 7])
    dif7[dif7>5] = 5
    dif7[dif7<-5] = -5
    dif7 = (dif7 - np.min(dif7)) / (np.max(dif7) - np.min(dif7))

    # med 11
    dif11 = median_filter(dif,[11, 11, 11])
    dif11[dif11>5] = 5
    dif11[dif11<-5] = -5
    dif11 = (dif11 - np.min(dif11)) / (np.max(dif11) - np.min(dif11))



# location feature
    niiname = im_root + imglst[ind] + '/locatProb.nii.gz'
    nii = nib.load(niiname)
    loc = nii.get_data()
    loc = loc.astype(np.float32)

    plabel = np.zeros((H, W, Z))  # label of an image

    probimg = np.zeros((H, W, Z))  # proba of an image

    idx_img = (im > 10) & (im < 50)

    plabel_re = plabel.reshape(H * W * Z, 1)
    probimg_re = probimg.reshape(H * W * Z, 1)

    mask_re = im.reshape(H * W * Z, 1)
    pidx = np.where((mask_re > 10) & (mask_re < 50))[0]
    im3_re = im3.reshape(H * W * Z, 1)
    im7_re = im7.reshape(H * W * Z, 1)
    im11_re = im11.reshape(H * W * Z, 1)
    dif3_re = dif3.reshape(H * W * Z, 1)
    dif7_re = dif7.reshape(H * W * Z, 1)
    dif11_re = dif11.reshape(H * W * Z, 1)
    dist_re = dist.reshape(H * W * Z, 1)
    loc_re = loc.reshape(H * W * Z, 1)
    csfprob_re = csfprob.reshape(H * W * Z, 1)

# feat reshape and classification one image at one time
    feat = np.hstack((im3_re[pidx], im7_re[pidx], im11_re[pidx], dif3_re[pidx], dif7_re[pidx], dif11_re[pidx], dist_re[pidx], loc_re[pidx],csfprob_re[pidx]))  # ravg[pidx],

    feat = feat[:, fidx]
    feat = np.nan_to_num(feat)

    res = clf.predict(feat)
    prop = clf.predict_proba(feat)

    plabel[idx_img] = res[:]
    probimg[idx_img] = prop[:, 1]

    # post-processing

    a = plabel[:int(H / 2), :, :]
    b = plabel[int(H / 2):H, :, :]

    an = np.sum(a == 1)

    bn = np.sum(b == 1)

    if an > bn:
        plabel[int(H / 2):H, :, :] = 0
    if bn > an:
        plabel[:int(H / 2), :, :] = 0

    plabel = plabel.astype(np.int)
    for k in range(0,Z):
        tmp = plabel[:,:,k]
        tmp = morphology.binary_fill_holes(tmp) #,structure=np.ones((5,5))
        tmp = morphology.binary_erosion(tmp)#,structure=np.ones((2,2))
        tmp = morphology.binary_dilation(tmp)#,structure=ndimage.generate_binary_structure(2,2)
        tmp = morphology.binary_fill_holes(tmp) #,structure=np.ones((5,5))
        plabel[:,:,k] = tmp

# save segmentation results
    savepath = 'test_example/' + imglst[ind]
    if os.path.isdir(savepath) == False:
        os.makedirs(savepath)

    array_img = nib.Nifti1Image(plabel, aff1, hdr1)
    labelname = savepath + '/Lesion_Seg_res.nii.gz'
    nib.save(array_img, labelname)


    end = time.time()
    print('complete the ' + str(ind) + 'th image! It took ' + str(end - start))




