import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np
from config.constants import METRICS_DECIMAL_PLACES, REPORT_OUTPUT

class ExcelReporter:
    def __init__(self, output_filepath: str = REPORT_OUTPUT):
        self.output_filepath = Path(output_filepath)
        self.detection_records_collection = []
        self.timing_records_collection = []

    def log_detection_record(self,
                            image_identifier: str,
                            image_filename_string: str,
                            stage1_result_binary: str,
                            bounding_boxes_list: List[List[int]],
                            confidence_score_list: List[float],
                            damage_type_string: str = None,
                            severity_score_float: float = None,
                            execution_time_dict: Dict[str, float] = None):
        if execution_time_dict is None:
            execution_time_dict = {}

        self.detection_records_collection.append({
            'ID': len(self.detection_records_collection) + 1,
            'Image_ID': image_identifier,
            'Image_Name': image_filename_string,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Stage1_Binary': stage1_result_binary,
            'Num_Detections': len(bounding_boxes_list),
            'BBox_Coordinates': str(bounding_boxes_list),
            'Confidence_Scores': str([round(confidence_val, METRICS_DECIMAL_PLACES) for confidence_val in confidence_score_list]),
            'Avg_Confidence': round(np.mean(confidence_score_list), METRICS_DECIMAL_PLACES) if confidence_score_list else 0,
            'Damage_Type': damage_type_string or 'N/A',
            'Severity_Score': round(severity_score_float, METRICS_DECIMAL_PLACES) if severity_score_float else 'N/A',
            'Alarm_Set': 'Yes' if severity_score_float and severity_score_float >= 7 else 'No',
            'Stage1_Time': round(execution_time_dict.get('stage1', 0), METRICS_DECIMAL_PLACES),
            'Stage2_Time': round(execution_time_dict.get('stage2', 0), METRICS_DECIMAL_PLACES),
            'Total_Execution_Time': round(execution_time_dict.get('total', 0), METRICS_DECIMAL_PLACES),
        })

    def log_timing_record(self, stage_name_string: str, execution_time_dict: Dict[str, float]):
        self.timing_records_collection.append({
            'Timestamp': datetime.now().isoformat(),
            'Stage': stage_name_string,
            'Preprocessing': round(execution_time_dict.get('preprocessing', 0), METRICS_DECIMAL_PLACES),
            'Inference': round(execution_time_dict.get('inference', 0), METRICS_DECIMAL_PLACES),
            'Postprocessing': round(execution_time_dict.get('postprocessing', 0), METRICS_DECIMAL_PLACES),
            'Total': round(execution_time_dict.get('total', 0), METRICS_DECIMAL_PLACES),
        })

    def generate_summary_sheet(self, model_metrics_dict: Dict) -> pd.DataFrame:
        summary_dataframe = pd.DataFrame([{
            'Stage': 'Stage 2: Damage Localization',
            'Model': model_metrics_dict.get('model_name', 'YOLOv8-Medium'),
            'Accuracy': model_metrics_dict.get('accuracy', 0),
            'Precision': model_metrics_dict.get('precision', 0),
            'Recall': model_metrics_dict.get('recall', 0),
            'F1_Score': model_metrics_dict.get('f1_score', 0),
            'mAP': model_metrics_dict.get('mAP', 0),
            'IoU': model_metrics_dict.get('IoU', 0),
            'Avg_Inference_Time (s)': model_metrics_dict.get('avg_inference_time', 0),
            'Time_Loss (s)': model_metrics_dict.get('time_loss', 0),
            'Num_Images_Tested': model_metrics_dict.get('num_images', 0),
            'Date': datetime.now().strftime('%Y-%m-%d'),
        }])
        return summary_dataframe

    def export_all_to_excel(self, model_metrics_dict: Dict = None, filename_override: str = None):
        if filename_override is None:
            final_filename = self.output_filepath
        else:
            final_filename = Path(filename_override)

        final_filename.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(final_filename, engine='openpyxl') as excel_writer:
            if self.detection_records_collection:
                detection_dataframe = pd.DataFrame(self.detection_records_collection)
                detection_dataframe.to_excel(excel_writer, sheet_name='Detection_Results', index=False)

            if model_metrics_dict:
                summary_dataframe = self.generate_summary_sheet(model_metrics_dict)
                summary_dataframe.to_excel(excel_writer, sheet_name='Summary_Metrics', index=False)

            if self.timing_records_collection:
                timing_dataframe = pd.DataFrame(self.timing_records_collection)
                timing_dataframe.to_excel(excel_writer, sheet_name='Timing_Analysis', index=False)

            if self.detection_records_collection:
                statistics_dataframe = self._build_statistics_sheet()
                statistics_dataframe.to_excel(excel_writer, sheet_name='Statistics', index=False)

    def _build_statistics_sheet(self) -> pd.DataFrame:
        detection_dataframe = pd.DataFrame(self.detection_records_collection)

        statistics_data_dict = {
            'Metric': [
                'Total_Images',
                'Damaged_Images',
                'Undamaged_Images',
                'Avg_Execution_Time',
                'Critical_Alarms',
                'Avg_Detections_Per_Image',
                'Avg_Confidence'
            ],
            'Value': [
                len(self.detection_records_collection),
                len(detection_dataframe[detection_dataframe['Stage1_Binary'] == 'Damaged']),
                len(detection_dataframe[detection_dataframe['Stage1_Binary'] == 'Not Damaged']),
                round(detection_dataframe['Total_Execution_Time'].mean(), METRICS_DECIMAL_PLACES),
                len(detection_dataframe[detection_dataframe['Alarm_Set'] == 'Yes']),
                round(detection_dataframe['Num_Detections'].mean(), METRICS_DECIMAL_PLACES),
                round(detection_dataframe['Avg_Confidence'].mean(), METRICS_DECIMAL_PLACES),
            ]
        }

        return pd.DataFrame(statistics_data_dict)
