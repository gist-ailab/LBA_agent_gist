# def analyze_cases(file1_path, file2_path):
#     # 텍스트파일1: filename -> (label1, label2, flag)
#     file1_dict = {}
#     normal_prediction_files = set()  # flag == 0
#     wrong_prediction_files = set()   # flag == 1
#     with open(file1_path, 'r') as f1:
#         for line in f1:
#             parts = line.strip().split(' ')
#             if len(parts) < 4:
#                 continue
#             filename, label1, label2, flag = parts
#             file1_dict[filename] = (label1, label2, flag)
#             if flag == '1':
#                 wrong_prediction_files.add(filename)
#             elif flag == '0':
#                 normal_prediction_files.add(filename)
#     # 텍스트파일2: filename -> (gt_label, noisy_label)
#     file2_dict = {}
#     with open(file2_path, 'r') as f2:
#         for line in f2:
#             parts = line.strip().split(' ')
#             if len(parts) < 3:
#                 continue
#             filename, gt_label, noisy_label = parts
#             file2_dict[filename] = gt_label

#     # case 카운트 초기화
#     case1_false_negative = 0        # flag=0인데 file2에 있음 (모델은 clean이라 했는데 실제로는 noisy)
#     case2_prediction_failure = 0    # label2 ≠ gt
#     case3_prediction_success = 0    # label2 == gt
#     case4_wrong_prediction = 0      # flag=1인데 file2에 없음
#     case5_clean_prediction = 0      # flag=0이고 file2에 없음 (정상 clean 판별)

#     # Case 1 처리 (False Negative)
#     case1_set = set()
#     for filename in normal_prediction_files:
#         if filename in file2_dict:
#             case1_false_negative += 1
#             case1_set.add(filename)

#     # Case 2, 3 처리 (예측 실패 / 성공)
#     for filename, gt_label in file2_dict.items():
#         if filename in file1_dict:
#             _, label2, _ = file1_dict[filename]
#             if label2 == gt_label:
#                 case3_prediction_success += 1
#             else:
#                 case2_prediction_failure += 1

#     # Case 4 처리 (False Positive)
#     for filename in wrong_prediction_files:
#         if filename not in file2_dict:
#             case4_wrong_prediction += 1

#     # Case 5 처리 (Clean Prediction)
#     for filename in normal_prediction_files:
#         if filename not in file2_dict:  # noise 파일 목록에 없고
#             case5_clean_prediction += 1  # flag도 0이므로 clean한 것으로 간주

#     # 결과 출력
#     print(f"Case 1 - False Negative (flag=0 but in file2): {case1_false_negative}")
#     print(f"Case 2 - Prediction Failure (label2 ≠ gt): {case2_prediction_failure}")
#     print(f"Case 3 - Prediction Success (label2 == gt): {case3_prediction_success}")
#     print(f"Case 4 - Wrong Prediction (flag=1 but missing in file2): {case4_wrong_prediction}")
#     print(f"Case 5 - Clean Prediction (flag=0 and not in file2): {case5_clean_prediction}")

# # 사용 예시
# analyze_cases(
#     '/home/work/Workspaces/yunjae_heo/LBA_third_molar_noise_detection/ham10000_train_data_noise_label_list.txt',
#     '/home/work/Workspaces/yunjae_heo/LBA_third_molar_noise_detection/noise_imgs.txt'
# )


def analyze_cases_with_confusion_matrix(file1_path, file2_path):
    # 파일1 읽기
    file1_dict = {}
    with open(file1_path, 'r') as f1:
        for line in f1:
            parts = line.strip().split(' ')
            if len(parts) < 4:
                continue
            filename, label1, label2, flag = parts
            file1_dict[filename] = (label1, label2, flag)

    # 파일2 읽기 (노이즈 목록)
    noise_set = set()
    with open(file2_path, 'r') as f2:
        for line in f2:
            parts = line.strip().split(' ')
            if len(parts) < 3:
                continue
            filename, _, _ = parts
            noise_set.add(filename)

    # confusion matrix 항목 초기화
    TP = FP = TN = FN = 0

    for filename, (_, _, flag) in file1_dict.items():
        is_pred_noisy = (flag == '1')
        is_actual_noisy = (filename in noise_set)

        if is_pred_noisy and is_actual_noisy:
            TP += 1
        elif is_pred_noisy and not is_actual_noisy:
            FP += 1
        elif not is_pred_noisy and is_actual_noisy:
            FN += 1
        elif not is_pred_noisy and not is_actual_noisy:
            TN += 1

    # 결과 출력
    print("Confusion Matrix (Noise Detection)")
    print("------------------------------")
    print(f"TP (Predicted Noisy & Actually Noisy):     {TP}")
    print(f"FP (Predicted Noisy & Actually Clean):     {FP}")
    print(f"FN (Predicted Clean & Actually Noisy):     {FN}")
    print(f"TN (Predicted Clean & Actually Clean):     {TN}")

    # Optional: 출력 형태를 표로
    print("\nFormatted Matrix:")
    print(f"{'':<25} {'Actual Clean':<20} {'Actual Noisy'}")
    print(f"{'Predicted Clean':<25} {TN:<20} {FN}")
    print(f"{'Predicted Noisy':<25} {FP:<20} {TP}")

# 사용 예시
analyze_cases_with_confusion_matrix(
    '/home/work/Workspaces/yunjae_heo/LBA_third_molar_noise_detection/ham10000_train_data_noise_label_list.txt',
    '/home/work/Workspaces/yunjae_heo/LBA_third_molar_noise_detection/noise_imgs.txt'
)
