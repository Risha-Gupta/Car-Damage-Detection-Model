import numpy as np
from typing import List, Tuple, Dict
from config.constants import METRICS_DECIMAL_PLACES

class DetectionMetrics:
    """Enhanced metrics for Detectron2 with COCO evaluation support."""
    
    @staticmethod
    def calculate_iou(box_one: np.ndarray, box_two: np.ndarray) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        intersection_x1 = max(box_one[0], box_two[0])
        intersection_y1 = max(box_one[1], box_two[1])
        intersection_x2 = min(box_one[2], box_two[2])
        intersection_y2 = min(box_one[3], box_two[3])
        
        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0
        
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        
        area_box_one = (box_one[2] - box_one[0]) * (box_one[3] - box_one[1])
        area_box_two = (box_two[2] - box_two[0]) * (box_two[3] - box_two[1])
        union_area = area_box_one + area_box_two - intersection_area
        
        iou_score = intersection_area / union_area if union_area > 0 else 0.0
        return round(iou_score, METRICS_DECIMAL_PLACES)

    @staticmethod
    def match_predictions(predictions: List[Tuple],
                         ground_truth_boxes: List[Tuple],
                         iou_threshold: float = 0.5) -> Tuple[int, int, int]:
        """Match predictions to ground truth with IoU threshold."""
        true_positives = 0
        false_positives = 0
        matched_ground_truth_indices = set()
        
        for predicted_box, predicted_confidence in predictions:
            best_iou_score = 0
            best_ground_truth_index = -1
            
            for gt_index, (gt_box, _) in enumerate(ground_truth_boxes):
                if gt_index in matched_ground_truth_indices:
                    continue
                
                iou_score = DetectionMetrics.calculate_iou(predicted_box, gt_box)
                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_ground_truth_index = gt_index
            
            if best_iou_score >= iou_threshold and best_ground_truth_index >= 0:
                true_positives += 1
                matched_ground_truth_indices.add(best_ground_truth_index)
            else:
                false_positives += 1
        
        false_negatives = len(ground_truth_boxes) - len(matched_ground_truth_indices)
        
        return true_positives, false_positives, false_negatives

    @staticmethod
    def calculate_precision_recall(true_positives: int, false_positives: int,
                                   false_negatives: int) -> Tuple[float, float]:
        """Calculate precision and recall with 4 decimal places."""
        precision_score = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall_score = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        return round(precision_score, METRICS_DECIMAL_PLACES), round(recall_score, METRICS_DECIMAL_PLACES)

    @staticmethod
    def calculate_f1_score(precision_value: float, recall_value: float) -> float:
        """Calculate F1 score."""
        if (precision_value + recall_value) == 0:
            return 0.0
        f1_value = 2 * (precision_value * recall_value) / (precision_value + recall_value)
        return round(f1_value, METRICS_DECIMAL_PLACES)

    @staticmethod
    def calculate_mean_iou(predictions: List[Tuple],
                          ground_truth_boxes: List[Tuple],
                          iou_threshold: float = 0.5) -> float:
        """Calculate mean IoU for all predictions."""
        iou_scores = []
        
        for predicted_box, _ in predictions:
            max_iou = 0
            for gt_box, _ in ground_truth_boxes:
                iou_score = DetectionMetrics.calculate_iou(predicted_box, gt_box)
                max_iou = max(max_iou, iou_score)
            
            if max_iou >= iou_threshold:
                iou_scores.append(max_iou)
        
        mean_iou_value = np.mean(iou_scores) if iou_scores else 0.0
        return round(mean_iou_value, METRICS_DECIMAL_PLACES)

    @staticmethod
    def calculate_map(all_predictions: List[List[Tuple]],
                     all_ground_truths: List[List[Tuple]],
                     iou_thresholds: List[float] = None) -> float:
        """Calculate mean Average Precision."""
        if iou_thresholds is None:
            iou_thresholds = [0.5]
        
        map_scores = []
        
        for threshold in iou_thresholds:
            tp_list = []
            fp_list = []
            confidence_scores = []
            
            for predictions, ground_truth in zip(all_predictions, all_ground_truths):
                sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
                
                tp, fp, fn = DetectionMetrics.match_predictions(
                    sorted_predictions, ground_truth, iou_threshold=threshold
                )
                
                for idx in range(tp + fp):
                    if idx < tp:
                        tp_list.append(1)
                        fp_list.append(0)
                    else:
                        tp_list.append(0)
                        fp_list.append(1)
                    
                    if sorted_predictions:
                        confidence_scores.append(sorted_predictions[idx][1])
            
            if len(tp_list) == 0:
                map_scores.append(0.0)
            else:
                tp_array = np.array(tp_list)
                fp_array = np.array(fp_list)
                tp_cumulative = np.cumsum(tp_array)
                fp_cumulative = np.cumsum(fp_array)
                
                recall_values = tp_cumulative / (len(tp_list) + np.sum(fp_array == 1))
                precision_values = tp_cumulative / (tp_cumulative + fp_cumulative)
                
                average_precision = np.trapz(precision_values, recall_values)
                map_scores.append(average_precision)
        
        mean_average_precision = np.mean(map_scores)
        return round(mean_average_precision, METRICS_DECIMAL_PLACES)

    @staticmethod
    def calculate_accuracy(true_positives: int, true_negatives: int,
                          false_positives: int, false_negatives: int) -> float:
        """Calculate overall accuracy."""
        total = true_positives + true_negatives + false_positives + false_negatives
        if total == 0:
            return 0.0
        accuracy = (true_positives + true_negatives) / total
        return round(accuracy, METRICS_DECIMAL_PLACES)

    @staticmethod
    def calculate_confusion_matrix(predictions: List[int], ground_truth: List[int],
                                   num_classes: int = 6) -> np.ndarray:
        """Generate confusion matrix for damage classes."""
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for pred, gt in zip(predictions, ground_truth):
            if 0 <= pred < num_classes and 0 <= gt < num_classes:
                matrix[gt, pred] += 1
        return matrix


class PerformanceTracker:
    """Enhanced performance tracking with time loss measurement."""
    
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'iou': [],
            'map': [],
            'time_loss': []
        }
        self.inference_times = []
        self.stage_times = {}

    def add_metrics(self, accuracy_value, precision_value, recall_value,
                    f1_value, iou_value, map_score, time_loss=0.0):
        """Add metrics with time loss tracking."""
        self.metrics['accuracy'].append(accuracy_value)
        self.metrics['precision'].append(precision_value)
        self.metrics['recall'].append(recall_value)
        self.metrics['f1_score'].append(f1_value)
        self.metrics['iou'].append(iou_value)
        self.metrics['map'].append(map_score)
        self.metrics['time_loss'].append(time_loss)

    def track_inference_time(self, stage_name: str, duration: float):
        """Track inference time per stage."""
        if stage_name not in self.stage_times:
            self.stage_times[stage_name] = []
        self.stage_times[stage_name].append(duration)
        self.inference_times.append(duration)

    def get_summary(self) -> dict:
        """Return comprehensive metrics summary."""
        summary = {
            'accuracy': round(np.mean(self.metrics['accuracy']), METRICS_DECIMAL_PLACES) if self.metrics['accuracy'] else 0,
            'precision': round(np.mean(self.metrics['precision']), METRICS_DECIMAL_PLACES) if self.metrics['precision'] else 0,
            'recall': round(np.mean(self.metrics['recall']), METRICS_DECIMAL_PLACES) if self.metrics['recall'] else 0,
            'f1_score': round(np.mean(self.metrics['f1_score']), METRICS_DECIMAL_PLACES) if self.metrics['f1_score'] else 0,
            'iou': round(np.mean(self.metrics['iou']), METRICS_DECIMAL_PLACES) if self.metrics['iou'] else 0,
            'map': round(np.mean(self.metrics['map']), METRICS_DECIMAL_PLACES) if self.metrics['map'] else 0,
            'avg_time_loss': round(np.mean(self.metrics['time_loss']), METRICS_DECIMAL_PLACES) if self.metrics['time_loss'] else 0,
            'avg_inference_time': round(np.mean(self.inference_times), METRICS_DECIMAL_PLACES) if self.inference_times else 0,
            'stage_breakdown': {stage: round(np.mean(times), METRICS_DECIMAL_PLACES) 
                              for stage, times in self.stage_times.items()}
        }
        return summary
