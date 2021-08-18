import torch

#https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/bbox_overlaps.py
def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    zero = torch.tensor(0).to(bboxes1.device)
    eps = torch.tensor(eps).to(bboxes1.device)

    bboxes1 = bboxes1.float()
    bboxes2 = bboxes2.float()
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols), dtype=torch.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows), dtype=torch.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = torch.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = torch.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = torch.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = torch.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = torch.maximum(x_end - x_start, zero) * torch.maximum(
            y_end - y_start, zero)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = torch.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


#https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/mean_ap.py
def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[torch.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_bboxes_ignore = gt_bboxes_ignore or torch.zeros((0,4)).to(gt_bboxes.device)
    gt_ignore_inds = torch.cat(
        (torch.zeros(gt_bboxes.shape[0], dtype=torch.bool),
         torch.ones(gt_bboxes_ignore.shape[0], dtype=torch.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = torch.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = torch.zeros((num_scales, num_dets), dtype=torch.float32)
    fp = torch.zeros((num_scales, num_dets), dtype=torch.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp.to(det_bboxes.device), fp.to(det_bboxes.device)

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    # for each det, which gt overlaps most with it
    ious_max, ious_argmax = ious.max(1)
    # sort all dets in descending order by scores
    sort_inds = torch.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = torch.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = torch.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp.to(det_bboxes.device), fp.to(det_bboxes.device)
