"""
Main processing pipeline for Stage 4 component classification.
Orchestrates ROI extraction, inference, and result logging.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path

from config import batch_size, device, conf_threshold, components, out_dir
from data_loader import Stage3Loader
from preprocessing import ROI
from model_inference import Model
from output_handler import Logger, Excel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(out_dir, 'logs', 'pipeline.log')),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class Pipeline:
    """End-to-end classification pipeline."""
    
    def __init__(self, stage3_file: str, model_file: str, img_dir: str, dev: str = device):
        """
        Initialize pipeline.
        
        Args:
            stage3_file: Path to Stage 3 results JSON
            model_file: Path to component model checkpoint
            img_dir: Directory containing images
            dev: Device ("cuda" or "cpu")
        """
        self.stage3 = Stage3Loader(stage3_file)
        self.roi = ROI()
        self.model = Model(path=model_file, device=dev, num_classes=len(components))
        self.img_dir = img_dir
        self.logger = Logger()
        
        log.info("Pipeline initialized")
    
    def process_img(self, name: str) -> List[Dict[str, Any]]:
        """
        Process single image and its Stage 3 detections.
        
        Args:
            name: Image filename
            
        Returns:
            List of detection results
        """
        start = time.time()
        results = []
        path = os.path.join(self.img_dir, name)
        
        if not os.path.exists(path):
            log.warning(f"Image not found: {path}")
            return results
        
        # Get Stage 3 detections
        dets = self.stage3.get(name)
        
        if not dets:
            log.debug(f"No detections for {name}")
            return results
        
        # Process each detection
        for i, det in enumerate(dets):
            try:
                box = self.stage3.box(det)
                damage = self.stage3.damage_type(det)
                stage3_score = self.stage3.score(det)
                
                # Extract and preprocess ROI
                roi_data = self.roi.process(path, box)
                if roi_data is None:
                    log.warning(f"Failed to preprocess ROI from {name}")
                    continue
                
                # Predict components
                comp_idx, comp_scores = self.model.predict(roi_data)
                
                # Filter by confidence
                valid = [
                    (components[idx], score) 
                    for idx, score in zip(comp_idx, comp_scores)
                    if score >= conf_threshold
                ]
                
                if valid:
                    comps, scores = zip(*valid)
                    elapsed = time.time() - start
                    
                    entry = {
                        'image': name,
                        'time': datetime.now().isoformat(),
                        'box': box,
                        'damage': damage,
                        'components': list(comps),
                        'scores': [float(s) for s in scores],
                        'elapsed': elapsed
                    }
                    results.append(entry)
                    self.logger.add(entry)
                
            except Exception as e:
                log.error(f"Error processing detection {i} in {name}: {str(e)}")
                continue
        
        return results
    
    def process_dataset(self, img_list: List[str]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process dataset in batches.
        
        Args:
            img_list: List of image filenames
            
        Returns:
            Tuple of (all results, summary stats)
        """
        log.info(f"Starting: {len(img_list)} images")
        
        all_results = []
        done = 0
        errors = 0
        start_time = time.time()
        
        # Process in batches
        for b_start in range(0, len(img_list), batch_size):
            b_end = min(b_start + batch_size, len(img_list))
            batch = img_list[b_start:b_end]
            
            log.info(f"Batch {b_start // batch_size + 1}: {len(batch)} images")
            
            for name in batch:
                try:
                    res = self.process_img(name)
                    all_results.extend(res)
                    done += 1
                except Exception as e:
                    log.error(f"Error on {name}: {str(e)}")
                    errors += 1
        
        total_time = time.time() - start_time
        
        stats = {
            'total_images': len(img_list),
            'processed': done,
            'errors': errors,
            'total_detections': len(all_results),
            'time_sec': total_time,
            'avg_per_image': total_time / max(done, 1)
        }
        
        log.info(f"Complete. Stats: {stats}")
        
        return all_results, stats
