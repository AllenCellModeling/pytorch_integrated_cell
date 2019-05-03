import skimage.measure as measure
import skimage.filters as filters
import numpy as np


def props2summary(props):

    summary_props = [
        "area",
        "bbox_area",
        "convex_area",
        "centroid",
        "equivalent_diameter",
        "euler_number",
        "extent",
        "filled_area",
        "major_axis_length",
        "max_intensity",
        "mean_intensity",
        "min_intensity",
        "minor_axis_length",
        "moments",
        "moments_central",
        "moments_hu",
        "moments_normalized",
        "orientation",
        "perimeter",
        "solidity",
        "weighted_centroid",
        "weighted_moments_central",
        "weighted_moments_hu",
        "weighted_moments_normalized",
    ]

    prop_new = {}

    for k in summary_props:
        prop_list = list()

        for prop in props:
            p = np.array(prop[k])
            prop_list.append(p)

        if len(prop_list) > 1:
            prop_stack = np.stack(prop_list, -1)

            prop_mean = np.mean(prop_stack, -1)
            prop_std = np.std(prop_stack, -1)

            prop_total = np.sum(prop_stack, -1)

        else:
            prop_stack = prop_list[0]

            prop_mean = prop_stack
            prop_std = np.zeros(prop_stack.shape)

        prop_new[k + "_total"] = prop_total
        prop_new[k + "_mean"] = prop_mean
        prop_new[k + "_std"] = prop_std

    prop_new["num_objs"] = len(props)

    return prop_new


def find_main_obj(im_bw):
    im_label = measure.label(im_bw > 0)

    ulabels = np.unique(im_label[im_label > 0])

    label_counts = np.zeros(len(ulabels))

    for i, label in enumerate(ulabels):
        label_counts[i] = np.sum(im_label == label)

    return im_label == ulabels[np.argmax(label_counts)]


def ch_feats(im_bw, bg_thresh=1e-2):

    im_bg_sub = im_bw > bg_thresh

    props = {}

    im_bg_sub = find_main_obj(im_bg_sub)

    main_props = measure.regionprops(im_bg_sub.astype("uint8"), im_bw)
    main_props = props2summary(main_props)

    for k in main_props:
        props["main_" + k] = main_props[k]

    thresh = filters.threshold_otsu(im_bw[im_bg_sub])
    im_obj = measure.label(im_bw > thresh)

    thresh_props = measure.regionprops(im_obj, im_bw)
    thresh_props = props2summary(thresh_props)

    for k in thresh_props:
        props["obj_" + k] = thresh_props[k]

    return props


def im2feats(im, ch_names, bg_thresh=1e-2):

    feats = {}
    for i, ch_name in enumerate(ch_names):
        im_ch = im[i]

        feats[ch_name] = ch_feats(im_ch, bg_thresh)

    return feats
