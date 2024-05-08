import csv
import matplotlib.pyplot as plt

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

# Function to read bounding box data from CSV
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x = int(row[0])
            y = int(row[1])
            width = int(row[2])
            height = int(row[3])
            data.append((x, y, width, height))
    return data

# Function to calculate average IoU
def calculate_average_iou(tracking_results, ground_truth):
    total_frames = len(ground_truth)
    total_iou = 0

    for i in range(total_frames):
        gt_box = ground_truth[i]
        track_box = tracking_results[i]
        total_iou += calculate_iou(gt_box, track_box)

    return total_iou / total_frames if total_frames != 0 else 0

# Function to visualize IoU distribution
def visualize_iou_distribution(tracking_results, ground_truth):
    iou_scores = []

    for i in range(len(ground_truth)):
        gt_box = ground_truth[i]
        track_box = tracking_results[i]
        iou_scores.append(calculate_iou(gt_box, track_box))

    plt.hist(iou_scores, bins=20, range=(0, 1), alpha=0.75)
    plt.xlabel('IoU Scores')
    plt.ylabel('Frequency')
    plt.title('IoU Distribution')
    plt.show()

# Function to evaluate tracking algorithm with dynamic thresholding
def evaluate_tracking_with_dynamic_threshold(tracking_file, ground_truth_file):
    tracking_results = read_csv(tracking_file)
    ground_truth = read_csv(ground_truth_file)

    best_threshold = 0
    best_f1_score = 0

    thresholds = [i * 0.1 for i in range(6, 8)]  # Example: Evaluating from 0.6 to 0.7

    for threshold in thresholds:
        successes = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(len(ground_truth)):
            gt_box = ground_truth[i]
            track_box = tracking_results[i]

            iou = calculate_iou(gt_box, track_box)

            if iou >= threshold:
                successes += 1
                true_positives += 1
            else:
                false_negatives += 1

        false_positives = len(ground_truth) - successes

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold

    # Use the best threshold to evaluate and generate final results
    successes = sum(1 for i in range(len(ground_truth)) if calculate_iou(ground_truth[i], tracking_results[i]) >= best_threshold)
    total_frames = len(ground_truth)
    accuracy = successes / total_frames
    average_iou = calculate_average_iou(tracking_results, ground_truth)
    visualize_iou_distribution(tracking_results, ground_truth)

    return {
        'best_threshold': best_threshold,
        'best_f1_score': best_f1_score,
        'precision': precision,
        'recall': recall,
        'success_rate': accuracy,
        'average_iou': average_iou
    }

# File paths for tracking results and ground truth
tracking_results_file = 'bounding_boxes.csv'
ground_truth_file = 'groundtruthdata.csv'

# Evaluate the tracking algorithm with dynamic thresholding
evaluation_results = evaluate_tracking_with_dynamic_threshold(tracking_results_file, ground_truth_file)
print(f"Best Threshold: {evaluation_results['best_threshold']}")
print(f"Best F1 Score: {evaluation_results['best_f1_score']:.2f}")
print(f"Precision: {evaluation_results['precision']:.2f}")
print(f"Recall: {evaluation_results['recall']:.2f}")
print(f"Success Rate: {evaluation_results['success_rate']:.2f}")
print(f"Average IoU: {evaluation_results['average_iou']:.2f}")
