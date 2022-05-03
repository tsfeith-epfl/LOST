import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
from PIL import Image
from einops import reduce, rearrange, repeat

import matplotlib.pyplot as plt

def visualize_predictions(image, pred, seed, scales, dims, vis_folder, im_name, plot_seed=False):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    w_featmap, h_featmap = dims

    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
    )

    # Plot the seed
    if plot_seed:
        s_ = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
        size_ = np.asarray(scales) / 2
        cv2.rectangle(
            image,
            (int(s_[1] * scales[1] - (size_[1] / 2)), int(s_[0] * scales[0] - (size_[0] / 2))),
            (int(s_[1] * scales[1] + (size_[1] / 2)), int(s_[0] * scales[0] + (size_[0] / 2))),
            (0, 255, 0), -1,
        )

    pltname = f"{vis_folder}/LOST_{im_name}.png"
    Image.fromarray(image).save(pltname)
    print(f"Predictions saved at {pltname}.")
    
def visualize_mask(img, mask, patch_size, type_, vis_folder, im_name):
    mask_up = mask.cpu().bool().numpy().copy()
    mask_up = mask_up.repeat(patch_size, axis=0).repeat(patch_size, axis=1)
    mask_up = np.stack((mask_up,)*3, axis=-1)
    img_masked = img.copy()
    if type_ == 'fg':
        img_masked[np.logical_not(mask_up)] = 0
    if type_ == 'bg':
        img_masked[mask_up] = 0
    pltname = f"{vis_folder}/SSOD_{im_name}_mask_{type_}.png"
    Image.fromarray(img_masked.astype('uint8')).save(pltname)    
    print(f"Predictions saved at {pltname}.")
    
def visualize_bbox(img, pred, gt_bboxs, vis_folder, im_name):
    cv2.rectangle(
        img,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
    )
    for gt_bbox in gt_bboxs:
        cv2.rectangle(
            img,
            (int(gt_bbox[0]), int(gt_bbox[1])),
            (int(gt_bbox[2]), int(gt_bbox[3])),
            (0, 255, 0), 3,
        )

    pltname = f"{vis_folder}/SSOD_{im_name}_bbox.png"
    Image.fromarray(img.astype('uint8')).save(pltname)
    print(f"Predictions saved at {pltname}.")
    
def visualize_fms(A, seed, scores, dims, scales, output_folder, im_name):
    """
    Visualization of the maps presented in Figure 2 of the paper. 
    """
    w_featmap, h_featmap = dims

    # Binarized similarity
    binA = A.copy()
    binA[binA < 0] = 0
    binA[binA > 0] = 1

    # Get binarized correlation for this pixel and make it appear in gray
    im_corr = np.zeros((3, len(scores)))
    where = binA[seed, :] > 0
    im_corr[:, where] = np.array([128 / 255, 133 / 255, 133 / 255]).reshape((3, 1))
    # Show selected pixel in green
    im_corr[:, seed] = [204 / 255, 37 / 255, 41 / 255]
    # Reshape and rescale
    im_corr = im_corr.reshape((3, w_featmap, h_featmap))
    im_corr = (
        nn.functional.interpolate(
            torch.from_numpy(im_corr).unsqueeze(0),
            scale_factor=scales,
            mode="nearest",
        )[0].cpu().numpy()
    )

    # Save correlations
    skimage.io.imsave(
        fname=f"{output_folder}/corr_{im_name}.png",
        arr=im_corr.transpose((1, 2, 0)),
    )
    print(f"Image saved at {output_folder}/corr_{im_name}.png .")

    # Save inverse degree
    im_deg = (
        nn.functional.interpolate(
            torch.from_numpy(1 / binA.sum(-1)).reshape(1, 1, w_featmap, h_featmap),
            scale_factor=scales,
            mode="nearest",
        )[0][0].cpu().numpy()
    )
    plt.imsave(fname=f"{output_folder}/deg_{im_name}.png", arr=im_deg)
    print(f"Image saved at {output_folder}/deg_{im_name}.png .")

def visualize_seed_expansion(image, pred, seed, pred_seed, scales, dims, vis_folder, im_name):
    """
    Visualization of the seed expansion presented in Figure 3 of the paper. 
    """
    w_featmap, h_featmap = dims

    # Before expansion
    cv2.rectangle(
        image,
        (int(pred_seed[0]), int(pred_seed[1])),
        (int(pred_seed[2]), int(pred_seed[3])),
        (204, 204, 0),  # Yellow
        3,
    )

    # After expansion
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (204, 0, 204),  # Magenta
        3,
    )

    # Position of the seed
    center = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
    start_1 = center[0] * scales[0]
    end_1 = center[0] * scales[0] + scales[0]
    start_2 = center[1] * scales[1]
    end_2 = center[1] * scales[1] + scales[1]
    image[start_1:end_1, start_2:end_2, 0] = 204
    image[start_1:end_1, start_2:end_2, 1] = 37
    image[start_1:end_1, start_2:end_2, 2] = 41

    pltname = f"{vis_folder}/LOST_seed_expansion_{im_name}.png"
    Image.fromarray(image).save(pltname)
    print(f"Image saved at {pltname}.")

def plot_2d_clustering(image, assignments, n_h, n_w, indices, patch_size, fig_name = None, show_fig = False):
    # Process the indices
    # indices = indices[0].argmax(dim=-1).squeeze().tolist()
    # indices = [(i // n_w, i % n_w) for i in indices]
    indices = torch.nonzero(indices[0]).tolist()
    indices_dict = {}
    for r in indices:
        if r[0] in indices_dict.keys():
            indices_dict[r[0]].append((r[1] // n_w, r[1] % n_w))
        else:
            indices_dict[r[0]] = [(r[1] // n_w, r[1] % n_w)]

    # Tile the image
    image = image.cuda()
    # image = (image - image.min()) / (image.max() - image.min())
    image = rearrange(image, 'c (m h) (n w) -> (m n) c h w', m=n_h, n=n_w)

    # Plot
    k = assignments.shape[-1]
    m_plot = int(math.sqrt(k))
    n_plot = math.ceil(k / m_plot)
    fig, axes = plt.subplots(m_plot, n_plot, figsize=(n_plot * 10, m_plot * 10))

    # Re-weight the images
    images_tile = torch.einsum('n c h w, n k -> k n c h w', image, assignments[0, :, :])
    images_tile = rearrange(images_tile, 'k (m n) c h w -> k c (m h) (n w)', m=n_h, n=n_w)

    # Iterate over each centroid
    for l in range(k):
        i = l // n_plot
        j = l % n_plot
        image_tile = images_tile[l]
        image_tile = (image_tile - image_tile.min()) / (image_tile.max() - image_tile.min())

        # Get the corresponding index
        rows_cols = [(r * patch_size, c * patch_size) for r, c in indices_dict[l]]

        # Plot the query patch
        if m_plot > 1:
            for r, c in rows_cols:
                axes[i, j].plot(c, r, '-', marker='o', color='red', lw=1, mec='k', mew=1, markersize=20)

            axes[i, j].imshow(image_tile.permute(1, 2, 0).cpu())
            axes[i, j].axis('off')
        else:
            for r, c in rows_cols:
                axes[j].plot(c, r, '-', marker='o', color='red', lw=1, mec='k', mew=1, markersize=20)

            axes[j].imshow(image_tile.permute(1, 2, 0).cpu())
            axes[j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    if fig_name == None:
        index = random.randint(0, 100)
        plt.savefig('figures/object_discovery_{}'.format(index))
    else:
        plt.savefig('figures/{}'.format(fig_name))
    
    if show_fig:
        plt.show()
    else:
        plt.close()
