import torch
import scipy
import scipy.ndimage

import numpy as np
from datasets import bbox_iou

def fix_img_size(img, mode, patch_size, torch_shape):
    if torch_shape:
        if mode == 'pad_zero':
            size_im = (
                img.shape[0],
                int(np.ceil(img.shape[1] / patch_size) * patch_size),
                int(np.ceil(img.shape[2] / patch_size) * patch_size),
            )
            padded = np.zeros(size_im)
            padded[:, : img.shape[1], : img.shape[2]] = img
            img = padded
        elif mode == 'crop_topleft':
            w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
            img = img[:, :w, :h]
        elif mode == 'crop_center':
            w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
            dw, dh = img.shape[1] - w, img.shape[2] - h
            img = img[:, int(dw/2):int(w+dw/2), int(dh/2):int(h+dh/2)]
        elif mode == 'pad_mirror':
            pad_w, pad_h = patch_size - img.shape[1] % patch_size, patch_size - img.shape[2] % patch_size
            img = np.pad(img, ((0,0), (math.floor(pad_w/2), math.ceil(pad_w/2)), (math.floor(pad_h/2), math.ceil(pad_h/2))), 'symmetric')

    else:
        if mode == 'pad_zero':
            size_im = (
                int(np.ceil(img.shape[0] / patch_size) * patch_size),
                int(np.ceil(img.shape[1] / patch_size) * patch_size),
                img.shape[2]
            )
            padded = np.zeros(size_im)
            padded[: img.shape[0], : img.shape[1], :] = img
            img = padded
        elif mode == 'crop_topleft':
            w, h = img.shape[0] - img.shape[0] % patch_size, img.shape[1] - img.shape[1] % patch_size
            img = img[:w, :h, :]
        elif mode == 'crop_center':
            w, h = img.shape[0] - img.shape[0] % patch_size, img.shape[1] - img.shape[1] % patch_size
            dw, dh = img.shape[0] - w, img.shape[1] - h
            img = img[int(dw/2):int(w+dw/2), int(dh/2):int(h+dh/2), :]
        elif mode == 'pad_mirror':
            pad_w, pad_h = patch_size - img.shape[0] % patch_size, patch_size - img.shape[1] % patch_size
            img = np.pad(img, ((math.floor(pad_w/2), math.ceil(pad_w/2)), (math.floor(pad_h/2), math.ceil(pad_h/2)), (0,0)), 'symmetric')
            
    return img

def lost(feats, dims, scales, init_image_size, k_patches=100):
    """
    Implementation of LOST method.
    Inputs
        feats: the pixel/patche features of an image
        dims: dimension of the map from which the features are used
        scales: from image to map scale
        init_image_size: size of the image
        k_patches: number of k patches retrieved that are compared to the seed at seed expansion
    Outputs
        pred: box predictions
        A: binary affinity matrix
        scores: lowest degree scores for all patches
        seed: selected patch corresponding to an object
    """
    # Compute the similarity
    A = (feats @ feats.transpose(1, 2)).squeeze()

    # Compute the inverse degree centrality measure per patch
    sorted_patches, scores = patch_scoring(A)

    # Select the initial seed
    seed = sorted_patches[0]

    # Seed expansion
    potentials = sorted_patches[:k_patches]
    similars = potentials[A[seed, potentials] > 0.0]
    M = torch.sum(A[similars, :], dim=0)

    # Box extraction
    pred, _ = detect_box(
        M, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )

    return np.asarray(pred), A, scores, seed


def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


def detect_box(A, seed, dims, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    # Should not happen with LOST
    if cc == 0:
        raise ValueError("The seed is in the background component.")

    # Find box
    mask = np.where(labeled_array == cc)
    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    # Coordinate predictions for the feature space
    # Axis different then in image space
    pred_feats = [ymin, xmin, ymax, xmax]

    return pred, pred_feats


def dino_seg(attn, dims, patch_size, head=0):
    """
    Extraction of boxes based on the DINO segmentation method proposed in https://github.com/facebookresearch/dino. 
    """
    w_featmap, h_featmap = dims
    nh = attn.shape[1]
    official_th = 0.6

    # We keep only the output patch attention
    # Get the attentions corresponding to [CLS] token
    attentions = attn[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - official_th)
    idx2 = torch.argsort(idx)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    # Connected components
    labeled_array, num_features = scipy.ndimage.label(th_attn[head].cpu().numpy())

    # Find the biggest component
    size_components = [np.sum(labeled_array == c) for c in range(np.max(labeled_array))]

    if len(size_components) > 1:
        # Select the biggest component avoiding component 0 corresponding to background
        biggest_component = np.argmax(size_components[1:]) + 1
    else:
        # Cases of a single component
        biggest_component = 0

    # Mask corresponding to connected component
    mask = np.where(labeled_array == biggest_component)

    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image
    r_xmin, r_xmax = xmin * patch_size, xmax * patch_size
    r_ymin, r_ymax = ymin * patch_size, ymax * patch_size
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    return pred
