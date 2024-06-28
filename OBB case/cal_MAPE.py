from math import sqrt
import numpy as np
import os, glob


prediction_folder = '/workspace/yolov5_obb/runs/val/nonagnostic/xml3_batch32_iou/labels'
gt_folder = '/workspace/yolov5_obb/dataset3/val/labelTxt'



def count_hairs_pred(txt_file):
    count = 0
    with open(txt_file, 'r') as file:
        for line in file:
            class_id = int(line.split()[0])
            count += class_id+1  
    return count

def count_hairs_gt(txt_file):
    count = 0
    with open(txt_file, 'r') as file:
        for line in file:
            # import ipdb; ipdb.set_trace()
            class_id = int(line.split()[8])
            count += class_id+1 
    return count

def count_follicles(txt_file):
    with open(txt_file, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count

def cal_hair_error_rate(predicted, actual):
    if actual == 0:
        return 0  
    return abs(predicted - actual) / actual * 100

pred_counts = {}
for txt_file in glob.glob(os.path.join(prediction_folder, '*.txt')):
    file_name = os.path.basename(txt_file)
    pred_counts[file_name] = [count_hairs_pred(txt_file)]
    pred_counts[file_name].append(count_follicles(txt_file))

gt_counts = {}
for txt_file in glob.glob(os.path.join(gt_folder, '*.txt')):
    file_name = os.path.basename(txt_file)
    gt_counts[file_name] = [count_hairs_gt(txt_file)]
    gt_counts[file_name].append(count_follicles(txt_file))

Z = 2.58
n = 162

gt_follicles_list = []
yhat_follicles_list = []
gt_hairs_list = []
yhat_hairs_list = []

for file_name in gt_counts:
    if file_name in pred_counts:
        gt_follicles_list.append(gt_counts[file_name][1])
        yhat_follicles_list.append(pred_counts[file_name][1])
        gt_hairs_list.append(gt_counts[file_name][0])
        yhat_hairs_list.append(pred_counts[file_name][0])
               
gt_follicles_np = np.array(gt_follicles_list)
yhat_follicles_np = np.array(yhat_follicles_list)
gt_hairs_np = np.array(gt_hairs_list)
yhat_hairs_np = np.array(yhat_hairs_list)

MAE = np.mean((np.abs(np.subtract(gt_follicles_np, yhat_follicles_np))))
MAPE = np.mean((np.abs(np.subtract(gt_follicles_np, yhat_follicles_np)/ gt_follicles_np))) * 100
std = np.std((np.abs(np.subtract(gt_follicles_np, yhat_follicles_np)/ gt_follicles_np))) * 100
ci = Z * (std/sqrt(n))
print(f"total number of follicles: gt {gt_follicles_np.sum()} / pred {yhat_follicles_np.sum()}")
print(f"Mean Absolute Error: {MAE:.2f}")
print('Mean Absolute Percentage Error (MAPE) of hair follicles: ' + str(np.round(MAPE, 2)) + ' %')
print(f"{(100-MAPE):.2f}%")
print(f"STD: {std}")
print(f"CI: {str(np.round(MAPE, 2))}±{ci:.2f} (%)")

MAE = np.mean((np.abs(np.subtract(gt_hairs_np, yhat_hairs_np)))) 
MAPE = np.mean((np.abs(np.subtract(gt_hairs_np, yhat_hairs_np)/ gt_hairs_np))) * 100
std = np.std((np.abs(np.subtract(gt_hairs_np, yhat_hairs_np)/ gt_hairs_np))) * 100
ci = Z * (std/sqrt(n))
print(f"total number of hairs: gt {gt_hairs_np.sum()} / pred {yhat_hairs_np.sum()}")
print(f"Mean Absolute Error: {MAE:.2f}")
print('Mean Absolute Percentage Error (MAPE) hairs: ' + str(np.round(MAPE, 2)) + ' %')
print(f"{(100-MAPE):.2f}%")
print(f"STD: {std}")
print(f"CI: {str(np.round(MAPE, 2))}±{ci:.2f} (%)")