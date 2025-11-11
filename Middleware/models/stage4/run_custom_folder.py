"""
Inference script for custom image folders.
Run classification on any folder of images.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

from pipeline import Pipeline
from output_handler import Excel
from config import model_path, stage3_file, out_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Stage 4 classification on custom image folder"
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder with images'
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
    
    args = parser.parse_args()
    
    # Validate folder
    if not os.path.isdir(args.folder):
        log.error(f"Not found: {args.folder}")
        return 1
    
    # Get images
    formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    files = [f for f in os.listdir(args.folder) if f.lower().endswith(formats)]
    
    if not files:
        log.error(f"No images in {args.folder}")
        return 1
    
    log.info(f"Found {len(files)} images in {args.folder}")
    
    # Run
    try:
        pipe = Pipeline(
            stage3_file=args.stage3,
            model_file=args.model,
            img_dir=args.folder
        )
        
        results, stats = pipe.process_dataset(sorted(files))
        
        # Export
        df = pipe.logger.get_df()
        if not df.empty:
            out_file = os.path.join(args.output, 'results_custom.xlsx')
            Excel.save(df, out_file)
            print(f"\nComplete! Results in {out_file}")
        else:
            print("\nNo results")
        
        return 0
        
    except Exception as e:
        log.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
