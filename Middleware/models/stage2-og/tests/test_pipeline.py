import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import ImagePreprocessor, DataAugmentor, DataSplitter
from models import YOLOv8Detector, DetectionMetrics, PerformanceTracker
from logging import PerformanceLogger, ExcelReporter
from inference.predict import Stage2Predictor, PipelineIntegration

class TestSuite:
    @staticmethod
    def test_preprocessing():
        print("\n" + "="*60)
        print("[TEST] Image Preprocessing")
        print("="*60)

        preprocessor = ImagePreprocessor(target_size=(512, 512))
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        processed = preprocessor.process(dummy_image)

        assert processed.shape == (512, 512, 3), f"Expected (512, 512, 3), got {processed.shape}"
        assert processed.dtype == np.float32, f"Expected float32, got {processed.dtype}"
        assert processed.min() >= 0.0 and processed.max() <= 1.0, "Values not normalized to [0, 1]"

        print("✓ Preprocessing pipeline: PASSED")
        return True

    @staticmethod
    def test_augmentation():
        print("\n" + "="*60)
        print("[TEST] Data Augmentation")
        print("="*60)

        augmentor = DataAugmentor()
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        bboxes = [[100, 100, 200, 200], [300, 300, 400, 400]]

        aug_image, aug_bboxes = augmentor.augment(image, bboxes)

        assert aug_image.shape == image.shape, "Augmented image shape mismatch"
        assert len(aug_bboxes) == len(bboxes), "Bbox count mismatch"

        print(f"✓ Augmentation applied: {len(aug_bboxes)} bboxes preserved")
        print("✓ Augmentation pipeline: PASSED")
        return True

    @staticmethod
    def test_data_split():
        print("\n" + "="*60)
        print("[TEST] Data Splitting")
        print("="*60)

        images = [np.random.rand(512, 512, 3) for _ in range(100)]
        bboxes_list = [[[100, 100, 200, 200]] for _ in range(100)]

        splits = DataSplitter.split_data(images, bboxes_list, 0.7, 0.15)

        train_size = len(splits['train'][0])
        val_size = len(splits['val'][0])
        test_size = len(splits['test'][0])
        total_size = train_size + val_size + test_size

        assert train_size == 70, f"Expected 70 train samples, got {train_size}"
        assert val_size == 15, f"Expected 15 val samples, got {val_size}"
        assert test_size == 15, f"Expected 15 test samples, got {test_size}"
        assert total_size == 100, f"Expected 100 total, got {total_size}"

        print(f"✓ Train: {train_size}, Val: {val_size}, Test: {test_size}")
        print("✓ Data splitting: PASSED")
        return True

    @staticmethod
    def test_model_initialization():
        print("\n" + "="*60)
        print("[TEST] Model Initialization & Transfer Learning")
        print("="*60)

        model = YOLOv8Detector(model_size='n', pretrained=True)

        model.freeze_backbone(freeze_percentage=0.75)
        status = model.get_frozen_status()

        print(f"  Frozen: {status['frozen']}")
        print(f"  Trainable: {status['trainable']}")
        print(f"  Trainable %: {status['trainable_percentage']:.1f}%")

        assert status['trainable'] < status['frozen'], \
            "Trainable layers should be less than frozen"

        print("✓ Model initialization: PASSED")
        return True

    @staticmethod
    def test_metrics():
        print("\n" + "="*60)
        print("[TEST] Evaluation Metrics")
        print("="*60)

        box_first = np.array([100, 100, 200, 200])
        box_second = np.array([150, 150, 250, 250])

        iou_value = DetectionMetrics.calculate_iou(box_first, box_second)

        assert 0 <= iou_value <= 1, f"IoU should be between 0 and 1, got {iou_value}"
        print(f"  IoU between boxes: {iou_value:.4f}")

        true_pos, false_pos, false_neg = 8, 2, 1
        precision_val, recall_val = DetectionMetrics.calculate_precision_recall(true_pos, false_pos, false_neg)
        f1_val = DetectionMetrics.calculate_f1_score(precision_val, recall_val)

        assert 0 <= precision_val <= 1, "Precision out of range"
        assert 0 <= recall_val <= 1, "Recall out of range"
        assert 0 <= f1_val <= 1, "F1 out of range"

        print(f"  Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}")
        print("✓ Metrics calculation: PASSED")
        return True

    @staticmethod
    def test_logging():
        print("\n" + "="*60)
        print("[TEST] Logging & Reporting")
        print("="*60)

        logger = PerformanceLogger()
        reporter = ExcelReporter()

        logger.log_preprocessing_time(100, 2.34)
        reporter.log_detection_result(
            image_identifier='001',
            image_filename='car_001.jpg',
            binary_classification_result='Damaged',
            bounding_boxes=[[100, 100, 200, 200]],
            detection_confidence_scores=[0.95],
            severity_rating=6.5
        )

        print(f"  Logger initialized: OK")
        print(f"  Reporter records: {len(reporter.detection_result_records)}")

        print("✓ Logging & reporting: PASSED")
        return True

    @staticmethod
    def run_all_tests():
        print("\n\n" + "="*60)
        print("STAGE 2 - COMPREHENSIVE TEST SUITE")
        print("="*60)

        test_functions = [
            TestSuite.test_preprocessing,
            TestSuite.test_augmentation,
            TestSuite.test_data_split,
            TestSuite.test_model_initialization,
            TestSuite.test_metrics,
            TestSuite.test_logging,
        ]

        passed_count = 0
        failed_count = 0

        for test_function in test_functions:
            try:
                result = test_function()
                if result:
                    passed_count += 1
            except Exception as error:
                print(f"✗ TEST FAILED: {str(error)}")
                failed_count += 1

        print("\n" + "="*60)
        print(f"TEST RESULTS: {passed_count} PASSED, {failed_count} FAILED")
        print("="*60 + "\n")

        return failed_count == 0


if __name__ == '__main__':
    success = TestSuite.run_all_tests()
    sys.exit(0 if success else 1)
