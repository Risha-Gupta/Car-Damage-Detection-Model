import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
from config.constants import METRICS_DECIMAL_PLACES, LOG_DIR

class PerformanceLogger:
    def __init__(self, log_dir: str = LOG_DIR):
        self.log_directory = Path(log_dir)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.recorded_logs = {
            'training': {},
            'validation': {},
            'inference': {},
            'preprocessing': {},
            'postprocessing': {},
            'stage_timings': []
        }

        self.recorded_start_times = {}

    def start_training_epoch(self, epoch_number: int):
        self.recorded_start_times[f'epoch_{epoch_number}'] = time.time()

    def end_training_epoch(self, epoch_number: int, training_loss: float, validation_loss: float):
        elapsed_time = time.time() - self.recorded_start_times.get(f'epoch_{epoch_number}', time.time())

        self.recorded_logs['training'][epoch_number] = {
            'timestamp': datetime.now().isoformat(),
            'train_loss': round(training_loss, METRICS_DECIMAL_PLACES),
            'val_loss': round(validation_loss, METRICS_DECIMAL_PLACES),
            'epoch_time': round(elapsed_time, METRICS_DECIMAL_PLACES)
        }

    def log_training_batch(self, epoch_number: int, batch_number: int,
                          batch_loss: float, batch_time: float):
        if epoch_number not in self.recorded_logs['training']:
            self.recorded_logs['training'][epoch_number] = {'batches': []}

        if isinstance(self.recorded_logs['training'][epoch_number], dict) and 'batches' not in self.recorded_logs['training'][epoch_number]:
            self.recorded_logs['training'][epoch_number]['batches'] = []

        self.recorded_logs['training'][epoch_number]['batches'].append({
            'batch': batch_number,
            'loss': round(batch_loss, METRICS_DECIMAL_PLACES),
            'time': round(batch_time, METRICS_DECIMAL_PLACES)
        })

    def start_inference(self, image_identifier: str):
        self.recorded_start_times[f'inference_{image_identifier}'] = time.time()

    def log_inference_stage(self, image_identifier: str, stage_name: str, duration_seconds: float):
        inference_key = f'inference_{image_identifier}'

        if inference_key not in self.recorded_logs['inference']:
            self.recorded_logs['inference'][inference_key] = {
                'timestamp': datetime.now().isoformat(),
                'stages': {}
            }

        self.recorded_logs['inference'][inference_key]['stages'][stage_name] = round(duration_seconds, METRICS_DECIMAL_PLACES)

    def end_inference(self, image_identifier: str):
        elapsed_time = time.time() - self.recorded_start_times.get(f'inference_{image_identifier}', time.time())

        inference_key = f'inference_{image_identifier}'
        if inference_key in self.recorded_logs['inference']:
            self.recorded_logs['inference'][inference_key]['total_time'] = round(elapsed_time, METRICS_DECIMAL_PLACES)

    def log_preprocessing_time(self, image_count: int, total_time_seconds: float):
        self.recorded_logs['preprocessing'][len(self.recorded_logs['preprocessing'])] = {
            'timestamp': datetime.now().isoformat(),
            'image_count': image_count,
            'total_time': round(total_time_seconds, METRICS_DECIMAL_PLACES),
            'avg_time_per_image': round(total_time_seconds / image_count, METRICS_DECIMAL_PLACES)
        }

    def log_full_pipeline(self, image_name: str, stage_timings: Dict[str, float]):
        total_pipeline_time = sum(v for k, v in stage_timings.items() if k != 'total')
        time_loss_amount = stage_timings.get('total', total_pipeline_time) - sum(
            v for k, v in stage_timings.items()
            if k not in ['total', 'stage1_binary']
        )

        self.recorded_logs['stage_timings'].append({
            'timestamp': datetime.now().isoformat(),
            'image': image_name,
            'timings': {k: round(v, METRICS_DECIMAL_PLACES) for k, v in stage_timings.items()},
            'time_loss': round(time_loss_amount, METRICS_DECIMAL_PLACES),
            'time_loss_percentage': round((time_loss_amount / total_pipeline_time * 100), 2) if total_pipeline_time > 0 else 0
        })

    def calculate_training_stats(self) -> dict:
        if not self.recorded_logs['training']:
            return {}

        epoch_duration_list = [v['epoch_time'] for v in self.recorded_logs['training'].values()
                       if isinstance(v, dict) and 'epoch_time' in v]

        if not epoch_duration_list:
            return {}

        return {
            'total_epochs': len(epoch_duration_list),
            'total_time': round(sum(epoch_duration_list), METRICS_DECIMAL_PLACES),
            'avg_epoch_time': round(np.mean(epoch_duration_list), METRICS_DECIMAL_PLACES),
            'min_epoch_time': round(np.min(epoch_duration_list), METRICS_DECIMAL_PLACES),
            'max_epoch_time': round(np.max(epoch_duration_list), METRICS_DECIMAL_PLACES)
        }

    def calculate_inference_stats(self) -> dict:
        if not self.recorded_logs['inference']:
            return {}

        inference_durations = [v['total_time'] for v in self.recorded_logs['inference'].values()
                       if isinstance(v, dict) and 'total_time' in v]

        if not inference_durations:
            return {}

        return {
            'total_inferences': len(inference_durations),
            'avg_inference_time': round(np.mean(inference_durations), METRICS_DECIMAL_PLACES),
            'min_inference_time': round(np.min(inference_durations), METRICS_DECIMAL_PLACES),
            'max_inference_time': round(np.max(inference_durations), METRICS_DECIMAL_PLACES),
            'total_inference_time': round(sum(inference_durations), METRICS_DECIMAL_PLACES)
        }

    def identify_bottlenecks(self) -> dict:
        if not self.recorded_logs['stage_timings']:
            return {}

        stage_duration_mapping = {}

        for log_entry in self.recorded_logs['stage_timings']:
            for stage_name, timing_value in log_entry['timings'].items():
                if stage_name != 'total':
                    if stage_name not in stage_duration_mapping:
                        stage_duration_mapping[stage_name] = []
                    stage_duration_mapping[stage_name].append(timing_value)

        bottleneck_results = {}
        total_average_time = sum(np.mean(times) for times in stage_duration_mapping.values())

        for stage_name, duration_list in stage_duration_mapping.items():
            average_stage_time = np.mean(duration_list)
            bottleneck_results[stage_name] = {
                'avg_time': round(average_stage_time, METRICS_DECIMAL_PLACES),
                'percentage': round((average_stage_time / total_average_time * 100), 2)
            }

        return dict(sorted(bottleneck_results.items(), key=lambda x: x[1]['avg_time'], reverse=True))

    def export_json(self, filename: str = "performance_logs.json"):
        export_filepath = self.log_directory / filename

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'training_stats': self.calculate_training_stats(),
            'inference_stats': self.calculate_inference_stats(),
            'bottlenecks': self.identify_bottlenecks(),
            'raw_logs': self.recorded_logs
        }

        with open(export_filepath, 'w') as file_handle:
            json.dump(export_data, file_handle, indent=2)

    def get_performance_summary(self) -> str:
        summary_text = "\n" + "="*60
        summary_text += "\n[PERFORMANCE SUMMARY]"
        summary_text += "\n" + "="*60

        training_statistics = self.calculate_training_stats()
        if training_statistics:
            summary_text += f"\n\nTraining:"
            summary_text += f"\n  Total epochs: {training_statistics.get('total_epochs')}"
            summary_text += f"\n  Total time: {training_statistics.get('total_time')}s"
            summary_text += f"\n  Avg per epoch: {training_statistics.get('avg_epoch_time')}s"

        inference_statistics = self.calculate_inference_stats()
        if inference_statistics:
            summary_text += f"\n\nInference:"
            summary_text += f"\n  Total inferences: {inference_statistics.get('total_inferences')}"
            summary_text += f"\n  Avg time: {inference_statistics.get('avg_inference_time')}s per image"

        bottleneck_map = self.identify_bottlenecks()
        if bottleneck_map:
            summary_text += "\n\nBottlenecks (sorted by time):"
            for stage_name, stage_data in list(bottleneck_map.items())[:3]:
                summary_text += f"\n  {stage_name}: {stage_data['avg_time']}s ({stage_data['percentage']}%)"

        summary_text += "\n" + "="*60 + "\n"
        return summary_text
