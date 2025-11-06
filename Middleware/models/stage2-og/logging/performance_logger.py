import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
from config.constants import METRICS_DECIMAL_PLACES, LOG_DIR

class PerformanceLogger:
    def __init__(self, logging_directory: str = LOG_DIR):
        self.logging_directory = Path(logging_directory)
        self.logging_directory.mkdir(parents=True, exist_ok=True)

        self.activity_logs = {
            'training': {},
            'validation': {},
            'inference': {},
            'preprocessing': {},
            'postprocessing': {},
            'stage_timings': []
        }

        self.timer_registry = {}

    def start_training_epoch(self, epoch_number: int):
        self.timer_registry[f'epoch_{epoch_number}'] = time.time()

    def end_training_epoch(self, epoch_number: int, training_loss_value: float, validation_loss_value: float):
        elapsed_duration = time.time() - self.timer_registry.get(f'epoch_{epoch_number}', time.time())

        self.activity_logs['training'][epoch_number] = {
            'timestamp': datetime.now().isoformat(),
            'train_loss': round(training_loss_value, METRICS_DECIMAL_PLACES),
            'val_loss': round(validation_loss_value, METRICS_DECIMAL_PLACES),
            'epoch_time': round(elapsed_duration, METRICS_DECIMAL_PLACES)
        }

    def log_training_batch(self, epoch_number: int, batch_number: int,
                          batch_loss_value: float, batch_execution_time: float):
        if epoch_number not in self.activity_logs['training']:
            self.activity_logs['training'][epoch_number] = {'batches': []}

        if isinstance(self.activity_logs['training'][epoch_number], dict) and 'batches' not in self.activity_logs['training'][epoch_number]:
            self.activity_logs['training'][epoch_number]['batches'] = []

        self.activity_logs['training'][epoch_number]['batches'].append({
            'batch': batch_number,
            'loss': round(batch_loss_value, METRICS_DECIMAL_PLACES),
            'time': round(batch_execution_time, METRICS_DECIMAL_PLACES)
        })

    def start_inference_timer(self, image_identifier: str):
        self.timer_registry[f'inference_{image_identifier}'] = time.time()

    def log_inference_stage_time(self, image_identifier: str, stage_name_label: str, duration_seconds: float):
        inference_key = f'inference_{image_identifier}'

        if inference_key not in self.activity_logs['inference']:
            self.activity_logs['inference'][inference_key] = {
                'timestamp': datetime.now().isoformat(),
                'stages': {}
            }

        self.activity_logs['inference'][inference_key]['stages'][stage_name_label] = round(duration_seconds, METRICS_DECIMAL_PLACES)

    def end_inference_timer(self, image_identifier: str):
        elapsed_duration = time.time() - self.timer_registry.get(f'inference_{image_identifier}', time.time())

        inference_key = f'inference_{image_identifier}'
        if inference_key in self.activity_logs['inference']:
            self.activity_logs['inference'][inference_key]['total_time'] = round(elapsed_duration, METRICS_DECIMAL_PLACES)

    def log_preprocessing_metrics(self, image_count_processed: int, total_duration_seconds: float):
        self.activity_logs['preprocessing'][len(self.activity_logs['preprocessing'])] = {
            'timestamp': datetime.now().isoformat(),
            'image_count': image_count_processed,
            'total_time': round(total_duration_seconds, METRICS_DECIMAL_PLACES),
            'avg_time_per_image': round(total_duration_seconds / image_count_processed, METRICS_DECIMAL_PLACES)
        }

    def log_full_pipeline(self, image_filename: str, stage_timing_dict: Dict[str, float]):
        total_pipeline_duration = sum(duration_val for stage_key, duration_val in stage_timing_dict.items() if stage_key != 'total')
        time_overhead_amount = stage_timing_dict.get('total', total_pipeline_duration) - sum(
            duration_val for stage_key, duration_val in stage_timing_dict.items()
            if stage_key not in ['total', 'stage1_binary']
        )

        self.activity_logs['stage_timings'].append({
            'timestamp': datetime.now().isoformat(),
            'image': image_filename,
            'timings': {stage_key: round(duration_val, METRICS_DECIMAL_PLACES) for stage_key, duration_val in stage_timing_dict.items()},
            'time_loss': round(time_overhead_amount, METRICS_DECIMAL_PLACES),
            'time_loss_percentage': round((time_overhead_amount / total_pipeline_duration * 100), 2) if total_pipeline_duration > 0 else 0
        })

    def compute_training_statistics(self) -> dict:
        if not self.activity_logs['training']:
            return {}

        epoch_duration_list = [timing_data['epoch_time'] for timing_data in self.activity_logs['training'].values()
                       if isinstance(timing_data, dict) and 'epoch_time' in timing_data]

        if not epoch_duration_list:
            return {}

        return {
            'total_epochs': len(epoch_duration_list),
            'total_time': round(sum(epoch_duration_list), METRICS_DECIMAL_PLACES),
            'avg_epoch_time': round(np.mean(epoch_duration_list), METRICS_DECIMAL_PLACES),
            'min_epoch_time': round(np.min(epoch_duration_list), METRICS_DECIMAL_PLACES),
            'max_epoch_time': round(np.max(epoch_duration_list), METRICS_DECIMAL_PLACES)
        }

    def compute_inference_statistics(self) -> dict:
        if not self.activity_logs['inference']:
            return {}

        inference_duration_list = [timing_data['total_time'] for timing_data in self.activity_logs['inference'].values()
                       if isinstance(timing_data, dict) and 'total_time' in timing_data]

        if not inference_duration_list:
            return {}

        return {
            'total_inferences': len(inference_duration_list),
            'avg_inference_time': round(np.mean(inference_duration_list), METRICS_DECIMAL_PLACES),
            'min_inference_time': round(np.min(inference_duration_list), METRICS_DECIMAL_PLACES),
            'max_inference_time': round(np.max(inference_duration_list), METRICS_DECIMAL_PLACES),
            'total_inference_time': round(sum(inference_duration_list), METRICS_DECIMAL_PLACES)
        }

    def find_performance_bottlenecks(self) -> dict:
        if not self.activity_logs['stage_timings']:
            return {}

        stage_timing_aggregation = {}

        for log_entry in self.activity_logs['stage_timings']:
            for stage_name_label, timing_value in log_entry['timings'].items():
                if stage_name_label != 'total':
                    if stage_name_label not in stage_timing_aggregation:
                        stage_timing_aggregation[stage_name_label] = []
                    stage_timing_aggregation[stage_name_label].append(timing_value)

        bottleneck_analysis = {}
        total_average_duration = sum(np.mean(timing_list) for timing_list in stage_timing_aggregation.values())

        for stage_name_label, timing_list in stage_timing_aggregation.items():
            average_stage_duration = np.mean(timing_list)
            bottleneck_analysis[stage_name_label] = {
                'avg_time': round(average_stage_duration, METRICS_DECIMAL_PLACES),
                'percentage': round((average_stage_duration / total_average_duration * 100), 2)
            }

        return dict(sorted(bottleneck_analysis.items(), key=lambda entry: entry[1]['avg_time'], reverse=True))

    def export_to_json_file(self, filename: str = "performance_logs.json"):
        export_filepath = self.logging_directory / filename

        export_data_object = {
            'export_timestamp': datetime.now().isoformat(),
            'training_stats': self.compute_training_statistics(),
            'inference_stats': self.compute_inference_statistics(),
            'bottlenecks': self.find_performance_bottlenecks(),
            'raw_logs': self.activity_logs
        }

        with open(export_filepath, 'w') as file_obj:
            json.dump(export_data_object, file_obj, indent=2)

    def get_performance_summary(self) -> str:
        summary_message = "\n" + "="*60
        summary_message += "\n[PERFORMANCE SUMMARY]"
        summary_message += "\n" + "="*60

        training_stats_dict = self.compute_training_statistics()
        if training_stats_dict:
            summary_message += f"\n\nTraining:"
            summary_message += f"\n  Total epochs: {training_stats_dict.get('total_epochs')}"
            summary_message += f"\n  Total time: {training_stats_dict.get('total_time')}s"
            summary_message += f"\n  Avg per epoch: {training_stats_dict.get('avg_epoch_time')}s"

        inference_stats_dict = self.compute_inference_statistics()
        if inference_stats_dict:
            summary_message += f"\n\nInference:"
            summary_message += f"\n  Total inferences: {inference_stats_dict.get('total_inferences')}"
            summary_message += f"\n  Avg time: {inference_stats_dict.get('avg_inference_time')}s per image"

        bottleneck_dict = self.find_performance_bottlenecks()
        if bottleneck_dict:
            summary_message += "\n\nBottlenecks (sorted by time):"
            for stage_name_label, stage_metrics in list(bottleneck_dict.items())[:3]:
                summary_message += f"\n  {stage_name_label}: {stage_metrics['avg_time']}s ({stage_metrics['percentage']}%)"

        summary_message += "\n" + "="*60 + "\n"
        return summary_message
