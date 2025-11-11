"""
Output handling and result logging.
Manages Excel export and result formatting.
"""

import logging
import os
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from config import out_dir, excel_file, log_dir
from pathlib import Path

log = logging.getLogger(__name__)


class Logger:
    """Log detection results to storage and file."""
    
    def __init__(self):
        """Initialize logger."""
        self.results = []
        self.id_counter = 1
    
    def add(self, result: Dict[str, Any]) -> int:
        """
        Log a detection result.
        
        Args:
            result: Detection result dict
            
        Returns:
            Result ID
        """
        entry = {
            'id': self.id_counter,
            'image': result.get('image', ''),
            'time': result.get('time', ''),
            'box': str(result.get('box', ())),
            'damage': result.get('damage', ''),
            'components': ', '.join(result.get('components', [])),
            'scores': ', '.join([f"{s:.3f}" for s in result.get('scores', [])]),
            'elapsed': result.get('elapsed', 0.0)
        }
        
        self.results.append(entry)
        self.id_counter += 1
        
        return entry['id']
    
    def get_df(self) -> pd.DataFrame:
        """Get all results as DataFrame."""
        return pd.DataFrame(self.results)


class Excel:
    """Export results to Excel file."""
    
    @staticmethod
    def save(df: pd.DataFrame, out_path: str = excel_file):
        """
        Export results to Excel.
        
        Args:
            df: Results DataFrame
            out_path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Auto-fit columns
                ws = writer.sheets['Results']
                for col in ws.columns:
                    max_len = 0
                    col_cells = [cell for cell in col]
                    for cell in col_cells:
                        try:
                            if len(str(cell.value)) > max_len:
                                max_len = len(str(cell.value))
                        except:
                            pass
                    adj_width = min(max_len + 2, 50)
                    ws.column_dimensions[col_cells[0].column_letter].width = adj_width
            
            log.info(f"Saved: {out_path}")
            print(f"Results saved: {out_path}")
            
        except Exception as e:
            log.error(f"Excel export failed: {str(e)}")
            raise
    
    @staticmethod
    def summary(df: pd.DataFrame, out_path: str = None):
        """
        Export summary statistics.
        
        Args:
            df: Results DataFrame
            out_path: Output file path
        """
        if out_path is None:
            base = os.path.splitext(excel_file)[0]
            out_path = f"{base}_summary.xlsx"
        
        try:
            comp_counts = df['components'].value_counts()
            dmg_counts = df['damage'].value_counts()
            
            comp_df = pd.DataFrame({
                'component': comp_counts.index,
                'count': comp_counts.values
            })
            
            dmg_df = pd.DataFrame({
                'damage': dmg_counts.index,
                'count': dmg_counts.values
            })
            
            with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                comp_df.to_excel(writer, sheet_name='Components', index=False)
                dmg_df.to_excel(writer, sheet_name='Damages', index=False)
            
            log.info(f"Summary saved: {out_path}")
            
        except Exception as e:
            log.error(f"Summary export failed: {str(e)}")


class LogHandler:
    """Handle training and inference logging."""
    
    def __init__(self, output_dir='./output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_training_log(self, history_df, model_name='classifier'):
        """
        Save training metrics to Excel.
        Formats all numeric values to 4 decimal places.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = self.output_dir / f"training_log_{model_name}_{timestamp}.xlsx"
        
        # Format to 4 decimal places
        for col in history_df.select_dtypes(include=['float64']).columns:
            history_df[col] = history_df[col].round(4)
        
        history_df.to_excel(excel_path, index=False, sheet_name='Training')
        print(f"Training log saved: {excel_path}")
        
        return excel_path
    
    def save_inference_log(self, results, model_name='classifier'):
        """Save inference results to Excel with 4 decimal precision."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = self.output_dir / f"inference_log_{model_name}_{timestamp}.xlsx"
        
        df = pd.DataFrame(results)
        
        # Format numeric columns to 4 decimals
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].round(4)
        
        df.to_excel(excel_path, index=False, sheet_name='Inference')
        print(f"Inference log saved: {excel_path}")
        
        return excel_path
    
    def save_stage4_results(self, results, damage_type, roi_info, 
                           model_name='stage4'):
        """
        Save Stage 4 results with image, component, and damage info.
        Includes timing and confidence scores to 4 decimal places.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = self.output_dir / f"stage4_results_{model_name}_{timestamp}.xlsx"
        
        data = []
        for i, result in enumerate(results):
            row = {
                'ID': i + 1,
                'Image_Name': result.get('image', ''),
                'Timestamp': timestamp,
                'ROI': roi_info[i] if i < len(roi_info) else '',
                'Damage_Type': damage_type,
                'Component_Label': result.get('label', ''),
                'Component_Score': round(result.get('score', 0.0), 4),
                'Execution_Time': round(result.get('time', 0.0), 4)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(excel_path, index=False, sheet_name='Stage4')
        print(f"Stage 4 results saved: {excel_path}")
        
        return excel_path
