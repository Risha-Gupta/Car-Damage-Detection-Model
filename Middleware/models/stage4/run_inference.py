"""
Main inference script for running on dataset splits.
Use this to process train, val, or test images.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

from config import (
    train_imgs, val_imgs, test_imgs,
    stage3_file, model_path, out_dir, excel_file
)
from pipeline import Pipeline
from output_handler import Logger, Excel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def get_images(directory: str) -> list:
    """
    Get list of image files from directory.
    
    Args:
        directory: Path to images directory
        
    Returns:
        List of image filenames
    """
    if not os.path.exists(directory):
        log.error(f"Directory not found: {directory}")
        return []
    
    formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    files = [f for f in os.listdir(directory) if f.lower().endswith(formats)]
    
    log.info(f"Found {len(files)} images in {directory}")
    return sorted(files)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Stage 4 Component Classification on CarDD dataset"
    )
    parser.add_argument(
        '--split', 
        choices=['train', 'val', 'test'], 
        default='test',
        help='Dataset split to process (default: test)'
    )
    parser.add_argument(
        '--images',
        type=str,
        help='Custom images directory (overrides --split)'
    )
    parser.add_argument(
        '--stage3',
        type=str,
        default=stage3_file,
        help='Path to Stage 3 results JSON'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=model_path,
        help='Path to component model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=out_dir,
        help='Output directory'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max images to process'
    )
    
    args = parser.parse_args()
    
    # Determine image directory
    if args.images:
        img_dir = args.images
    else:
        img_dir = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }[args.split]
    
    log.info(f"Starting inference on {args.split}")
    log.info(f"Images: {img_dir}")
    log.info(f"Stage3: {args.stage3}")
    log.info(f"Model: {args.model}")
    
    # Get images
    img_list = get_images(img_dir)
    if not img_list:
        log.error("No images found")
        return 1
    
    # Limit if specified
    if args.limit:
        img_list = img_list[:args.limit]
        log.info(f"Limited to {len(img_list)} images")
    
    # Run pipeline
    try:
        pipe = Pipeline(
            stage3_file=args.stage3,
            model_file=args.model,
            img_dir=img_dir
        )
        
        # Process
        results, stats = pipe.process_dataset(img_list)
        
        # Log stats
        log.info(f"Stats: Total={stats['total_images']} Processed={stats['processed']} Errors={stats['errors']}")
        log.info(f"Detections={stats['total_detections']} Time={stats['time_sec']:.2f}s AvgPerImage={stats['avg_per_image']:.2f}s")
        
        # Export
        df = pipe.logger.get_df()
        if not df.empty:
            Excel.save(df, os.path.join(args.output, 'results.xlsx'))
            Excel.summary(df)
            print(f"\nComplete! Results in {args.output}")
        else:
            print("\nNo results to export")
        
        return 0
        
    except Exception as e:
        log.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
