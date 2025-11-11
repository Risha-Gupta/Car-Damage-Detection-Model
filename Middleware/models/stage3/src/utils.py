import shutil
import torch
import glob
import os
from pathlib import Path

def patch_torch_load():
    _original_load = torch.load
    
    def _patched_load(f, *args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_load(f, *args, **kwargs)
    
    torch.load = _patched_load
    print("✅ torch.load patched")

def save_checkpoint_callback(trainer, output_dir):
    """Save checkpoint after each epoch."""
    ckpt_path = trainer.last
    if ckpt_path and os.path.exists(ckpt_path):
        filename = os.path.basename(ckpt_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(ckpt_path, dest_path)
        print(f"💾 Saved: {dest_path}")