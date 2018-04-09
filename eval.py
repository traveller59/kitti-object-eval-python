import numpy as np
import numba
import io as sysio
from rotate_iou import rotate_iou_gpu_eval

@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


def compute_statistics(gt_anno,
                       dt_anno,
                       ignored_gt,
                       ignored_det,
                       dc_bboxes,
                       metric,
                       current_class,
                       thresh=0,
                       compute_fp=False,
                       compute_aos=False):
    MIN_OVERLAP = [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
    det_size = len(dt_anno["name"])
    gt_size = len(gt_anno["name"])
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if metric == 0:
        overlaps = image_box_overlap(dt_anno["bbox"], gt_anno["bbox"], -1)
    elif metric == 1:
        loc = gt_anno["location"][:, [0, 2]]
        dims = gt_anno["dimensions"][:, [0, 2]]
        rots = gt_anno["rotation_y"]
        gt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        loc = dt_anno["location"][:, [0, 2]]
        dims = dt_anno["dimensions"][:, [0, 2]]
        rots = dt_anno["rotation_y"]
        dt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        overlaps = bev_box_overlap(dt_boxes, gt_boxes, -1)
    if compute_fp:
        for i in range(det_size):
            if (dt_anno["score"][i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = []
    delta = []
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_anno["score"][j]
            if (not compute_fp
                    and (overlap > MIN_OVERLAP[metric][current_class])
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > MIN_OVERLAP[metric][current_class])
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > MIN_OVERLAP[metric][current_class])
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds.append(dt_anno["score"][det_idx])
            if compute_aos:
                delta.append(gt_anno["alpha"][i] - dt_anno["alpha"][det_idx])
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            if len(dc_bboxes) == 0:
                dc_bboxes = np.zeros([0, 4])
            else:
                dc_bboxes = np.stack(dc_bboxes, 0)
            overlaps_dt_dc = image_box_overlap(dt_anno["bbox"], dc_bboxes, 0)
            for i in range(len(dc_bboxes)):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j,
                                      i] > MIN_OVERLAP[metric][current_class]:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = [0] * fp
            for i in range(len(delta)):
                tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            assert len(tmp) == fp + tp
            assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds


@numba.jit(nopython=True)
def compute_statistics_v4(overlaps,
                          gt_datas,
                          dt_datas,
                          ignored_gt,
                          ignored_det,
                          dc_bboxes,
                          metric,
                          min_overlap,
                          thresh=0,
                          compute_fp=False,
                          compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]

@numba.jit(nopython=True)
def fused_compute_statistics(overlaps, pr, gt_nums, dt_nums, dc_nums, gt_datas,
                             dt_datas, dontcares, ignored_gts, ignored_dets,
                             metric, current_class, min_overlap, thresholds):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_v3(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                current_class=current_class,
                compute_fp=True)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]

def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num
        
def eval_class(gt_annos,
               dt_annos,
               current_class,
               difficulty,
               metric,
               min_overlap,
               num_parts=50):
    """Kitti eval. Only support 2d/bev/3d eval for now.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official: 
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]] 
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        precision: 1d array. precision for every sample PR point. 
    """
    assert len(gt_annos) == len(dt_annos)
    ignored_gts, ignored_dets, dontcares = [], [], []
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    total_num_valid_gt = 0
    thresholdss = []
    # start_time = time.time()
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    # print("DEBUG overlap time", time.time() - start_time)
    # start_time = time.time()
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    # print("DEBUG Clean time", time.time() - start_time)
    # start_time = time.time()

    for i in range(len(gt_annos)):
        rets = compute_statistics_v4(
            overlaps[i],
            gt_datas_list[i],
            dt_datas_list[i],
            # gt_annos[i],
            # dt_annos[i],
            ignored_gts[i],
            ignored_dets[i],
            dontcares[i],
            metric,
            min_overlap=min_overlap,
            thresh=0.0,
            compute_fp=False)
        tp, fp, fn, similarity, thresholds = rets
        thresholdss += thresholds.tolist()
    thresholdss = np.array(thresholdss)
    thresholds = get_thresholds(thresholdss, total_num_valid_gt)
    thresholds = np.array(thresholds)
    pr = np.zeros([len(thresholds), 4])
    # print("DEBUG calc stage1 time", time.time() - start_time)
    # start_time = time.time()
    idx = 0
    for j, num_part in enumerate(split_parts):
        gt_datas_part = np.concatenate(gt_datas_list[idx:idx + num_part], 0)
        dt_datas_part = np.concatenate(dt_datas_list[idx:idx + num_part], 0)
        dc_datas_part = np.concatenate(dontcares[idx:idx + num_part], 0)
        ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_part], 0)
        ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_part], 0)
        fused_compute_statistics(
            parted_overlaps[j],
            pr,
            total_gt_num[idx:idx + num_part],
            total_dt_num[idx:idx + num_part],
            total_dc_num[idx:idx + num_part],
            gt_datas_part,
            dt_datas_part,
            dc_datas_part,
            ignored_gts_part,
            ignored_dets_part,
            metric,
            min_overlap=min_overlap,
            thresholds=thresholds)
        idx += num_part

    # print("DEBUG other time", time.time() - start_time)
    N_SAMPLE_PTS = 41
    precision = np.zeros([N_SAMPLE_PTS])
    recall = np.zeros([N_SAMPLE_PTS])
    for i in range(len(thresholds)):
        recall[i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
        precision[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
    for i in range(len(thresholds)):
        precision[i] = np.max(precision[i:])

    return precision

def eval_class_v1(gt_annos, dt_annos, current_class, difficulty, metric):
    """same as eval_class but very slow. for debug and read. 
    """
    assert len(gt_annos) == len(dt_annos)
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    thresholdss = []

    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(ignored_gt)
        ignored_dets.append(ignored_det)
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        rets = compute_statistics(
            gt_annos[i],
            dt_annos[i],
            ignored_gt,
            ignored_det,
            dc_bboxes,
            metric,
            current_class=current_class,
            compute_fp=False)
        tp, fp, fn, similarity, thresholds = rets
        thresholdss += thresholds
    thresholdss = np.array(thresholdss)
    thresholds = get_thresholds(thresholdss, total_num_valid_gt)
    pr = np.zeros([len(thresholds), 4])
    for i in range(len(gt_annos)):
        for t in range(len(thresholds)):
            rets = compute_statistics(
                gt_annos[i],
                dt_annos[i],
                ignored_gts[i],
                ignored_dets[i],
                dontcares[i],
                metric,
                thresh=thresholds[t],
                current_class=current_class,
                compute_fp=True)
            tp, fp, fn, similarity, _ = rets
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
    N_SAMPLE_PTS = 41
    precision = np.zeros([N_SAMPLE_PTS])
    recall = np.zeros([N_SAMPLE_PTS])
    for i in range(len(thresholds)):
        recall[i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
        precision[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
    for i in range(len(thresholds)):
        precision[i] = np.max(precision[i:])

    return precision

def do_eval(gt_annos, dt_annos, current_class, min_overlap):
    mAP_bbox = []
    for i in range(3):  # i=difficulty
        prec = eval_class(gt_annos, dt_annos, current_class, i, 0,
                          min_overlap[0])
        mAP_bbox.append(get_mAP(prec))
    mAP_bev = []
    for i in range(3):
        prec = eval_class(gt_annos, dt_annos, current_class, i, 1,
                          min_overlap[1])
        mAP_bev.append(get_mAP(prec))
    mAP_3d = []
    for i in range(3):
        prec = eval_class(gt_annos, dt_annos, current_class, i, 2,
                          min_overlap[2])
        mAP_3d.append(get_mAP(prec))
    return mAP_bbox, mAP_bev, mAP_3d


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def get_official_eval_result(gt_annos, dt_annos, current_class):
    mAP_0_7 = np.array([[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]])
    mAP_0_5 = np.array([[0.7, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
    }
    mAPbbox, mAPbev, mAP3d = do_eval(gt_annos, dt_annos, current_class,
                                     mAP_0_7[:, current_class])

    result = print_str(
        (f"{class_to_name[current_class]} "
         "mAP@{:.2f}, {:.2f}, {:.2f}:".format(*mAP_0_7[:, current_class])))
    result += print_str((f"bbox mAP:{mAPbbox[0]:.2f}, "
                         f"{mAPbbox[1]:.2f}, "
                         f"{mAPbbox[2]:.2f}"))
    result += print_str((f"bev  mAP:{mAPbev[0]:.2f}, "
                         f"{mAPbev[1]:.2f}, "
                         f"{mAPbev[2]:.2f}"))
    result += print_str((f"3d   mAP:{mAP3d[0]:.2f}, "
                         f"{mAP3d[1]:.2f}, "
                         f"{mAP3d[2]:.2f}"))
    mAPbbox, mAPbev, mAP3d = do_eval(gt_annos, dt_annos, current_class,
                                     mAP_0_5[:, current_class])

    result += print_str(
        (f"{class_to_name[current_class]} "
         "mAP@{:.2f}, {:.2f}, {:.2f}:".format(*mAP_0_5[:, current_class])))
    result += print_str((f"bbox mAP:{mAPbbox[0]:.2f}, "
                         f"{mAPbbox[1]:.2f}, "
                         f"{mAPbbox[2]:.2f}"))
    result += print_str((f"bev  mAP:{mAPbev[0]:.2f}, "
                         f"{mAPbev[1]:.2f}, "
                         f"{mAPbev[2]:.2f}"))
    result += print_str((f"3d   mAP:{mAP3d[0]:.2f}, "
                         f"{mAP3d[1]:.2f}, "
                         f"{mAP3d[2]:.2f}"))

    return result
