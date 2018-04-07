# kitti-eval-python
Fast kitti eval in python(finish eval in less than 10 second)

## Usage
```Python
import kitti_common as kitti
from eval import eval_class
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
det_path = "/path/to/your_result_folder"
dt_annos = kitti.get_label_annos(det_path)
gt_path = "/path/to/your_gt_label_folder"
gt_split_file = "/path/to/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

def _get_mAP(prec):
    sums = 0
    for i in range(0, len(prec), 4):
        sums += prec[i]
    return sums / 11 * 100

mAP_0_7 = [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
print("mAP@0.7, 0.5, 0.5:")
prec_easy_bbox = eval_class(gt_annos, dt_annos, 0, 0, 0, mAP_0_7)
prec_mod_bbox = eval_class(gt_annos, dt_annos, 0, 1, 0, mAP_0_7)
prec_hard_bbox = eval_class(gt_annos, dt_annos, 0, 2, 0, mAP_0_7)
print((f"bbox mAP:{_get_mAP(prec_easy_bbox):.2f}, "
            f"{_get_mAP(prec_mod_bbox):.2f}, "
            f"{_get_mAP(prec_hard_bbox):.2f}"))

prec_easy_bev = eval_class(gt_annos, dt_annos, 0, 0, 1, mAP_0_7)
prec_mod_bev = eval_class(gt_annos, dt_annos, 0, 1, 1, mAP_0_7)
prec_hard_bev = eval_class(gt_annos, dt_annos, 0, 2, 1, mAP_0_7)
print((f"BEV mAP:{_get_mAP(prec_easy_bev):.2f}, "
            f"{_get_mAP(prec_mod_bev):.2f}, "
            f"{_get_mAP(prec_hard_bev):.2f}"))
prec_easy_3d = eval_class(gt_annos, dt_annos, 0, 0, 2, mAP_0_7)
prec_mod_3d = eval_class(gt_annos, dt_annos, 0, 1, 2, mAP_0_7)
prec_hard_3d = eval_class(gt_annos, dt_annos, 0, 2, 2, mAP_0_7)
print((f"3d mAP:{_get_mAP(prec_easy_3d):.2f}, "
            f"{_get_mAP(prec_mod_3d):.2f}, "
            f"{_get_mAP(prec_hard_3d):.2f}"))

```
