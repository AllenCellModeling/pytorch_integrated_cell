import numpy as np

from skimage.measure import label
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes
from aicsfeature.extractor import dna, cell


def im2feats(
    im_cell,
    im_nuc,
    im_structures,
    extra_features=["io_intensity", "bright_spots", "intensity", "skeleton", "texture"],
):
    # im_cell = Y x X x Z binary numpy array
    # im_nuc = Y x X x Z binary numpy array
    # im_structures = C x Y x X x Z numpy array

    # im_cell, im_nuc, im_structures, im_structures_seg, seg_cell, seg_nuc = im_process(
    #     im_cell, im_nuc, im_structures
    # )

    nuc_feats = dna.get_features(im_nuc, extra_features=extra_features)
    cell_feats = cell.get_features(im_cell, extra_features=extra_features)

    # feats_out = aicsfeature.kitchen_sink.kitchen_sink(
    #     im_cell=im_cell,
    #     im_nuc=im_nuc,
    #     im_structures=im_structures,
    #     seg_cell=seg_cell,
    #     seg_nuc=seg_nuc,
    #     extra_features=extra_features,
    # )
    # feats_out_seg = aicsfeature.kitchen_sink.kitchen_sink(
    #     im_cell=im_cell,
    #     im_nuc=im_nuc,
    #     im_structures=im_structures_seg,
    #     seg_cell=seg_cell,
    #     seg_nuc=seg_nuc,
    #     extra_features=extra_features,
    # )

    return [nuc_feats, cell_feats]


def im_process(im_cell, im_nuc, im_structures):
    seg_nuc = binary_fill_holes(im_nuc)
    seg_cell = binary_fill_holes(im_cell)
    seg_cell[seg_nuc] = 1

    im_nuc = (im_nuc * (255)).astype("uint16") * seg_nuc
    im_cell = (im_cell * (255)).astype("uint16") * seg_cell

    im_structures = [(im_ch * (255)).astype("uint16") for im_ch in im_structures]

    im_structures_seg = list()

    for i, im_structure in enumerate(im_structures):
        im_blur = gaussian(im_structure, 1)

        im_pix = im_structure[im_cell > 0]
        if np.all(im_pix == 0):
            im_structures_seg.append(im_structure)
            continue

        im_structures_seg.append(
            im_structure * (im_blur > threshold_otsu(im_blur[im_cell > 0]))
        )

    return im_cell, im_nuc, im_structures, im_structures_seg, seg_cell, seg_nuc


def find_main_obj(im_seg):
    im_label = label(im_seg)

    obj_index = -1
    max_cell_obj_size = -1
    for i in range(1, np.max(im_label) + 1):
        obj_size = np.sum(im_label == i)
        if obj_size > max_cell_obj_size:
            max_cell_obj_size = obj_size
            obj_index = i

    main_obj = im_label == obj_index

    return main_obj
