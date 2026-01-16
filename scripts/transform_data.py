#!/usr/bin/env python3
"""
LFW Dataset Structure Transformer

This script transforms the LFW (Labeled Faces in the Wild) dataset from its original
structure to the format expected by PyTorch's ImageFolder dataloader.

Original structure:
    data/lfw-deepfunneled/lfw-deepfunneled/
    ├── Aaron_Peirsol/
    │   ├── Aaron_Peirsol_0001.jpg
    │   ├── Aaron_Peirsol_0002.jpg
    │   └── ...
    ├── Adam_Sandler/
    │   └── ...
    
    data/peopleDevTrain.csv  (contains people names for training)
    data/peopleDevTest.csv   (contains people names for validation)

Expected output structure:
    data/train/
    ├── person_001/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── person_002/
    │   └── ...
    
    data/val/
    ├── person_001/
    │   ├── img1.jpg
    │   └── ...
    ├── person_002/
    │   └── ...
"""

import os
import shutil
import argparse
import csv
from pathlib import Path
from typing import Set


def read_people_from_csv(csv_path: str) -> Set[str]:
    """
    Read person names from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Set of person names
    """
    people = set()
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if row:  # Check if row is not empty
                people.add(row[0])  # First column is the name
    return people


def get_all_people_from_source(source_dir: str) -> Set[str]:
    """
    Get all person names from the source directory.
    
    Args:
        source_dir: Path to the source images directory
        
    Returns:
        Set of all person folder names
    """
    return {d for d in os.listdir(source_dir) 
            if os.path.isdir(os.path.join(source_dir, d))}


def transform_dataset(
    source_dir: str,
    output_dir: str,
    train_csv: str = None,
    val_csv: str = None,
    use_symlinks: bool = False,
    verbose: bool = True
):
    """
    Transform the LFW dataset to the expected dataloader format.
    
    Args:
        source_dir: Path to the source images directory (lfw-deepfunneled/lfw-deepfunneled)
        output_dir: Path to the output directory (will contain train/ and val/)
        train_csv: Path to peopleDevTrain.csv (optional)
        val_csv: Path to peopleDevTest.csv (optional)
        use_symlinks: If True, create symbolic links instead of copying files
        verbose: If True, print progress information
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Validate source directory
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get people for train and val sets
    if train_csv and val_csv:
        train_people = read_people_from_csv(train_csv)
        val_people = read_people_from_csv(val_csv)
        
        if verbose:
            print(f"Loaded {len(train_people)} people for training from {train_csv}")
            print(f"Loaded {len(val_people)} people for validation from {val_csv}")
    else:
        # If no CSV files provided, use all people for training
        all_people = get_all_people_from_source(source_dir)
        train_people = all_people
        val_people = set()
        
        if verbose:
            print(f"No CSV files provided. Using all {len(all_people)} people for training.")
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nCreated output directories:")
        print(f"  Train: {train_dir}")
        print(f"  Val: {val_dir}")
    
    # Process training data
    train_count = process_split(
        source_path, train_dir, train_people, 
        use_symlinks, verbose, "train"
    )
    
    # Process validation data
    val_count = process_split(
        source_path, val_dir, val_people,
        use_symlinks, verbose, "val"
    )
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Transformation complete!")
        print(f"  Training: {train_count['people']} people, {train_count['images']} images")
        print(f"  Validation: {val_count['people']} people, {val_count['images']} images")
        print(f"{'='*50}")


def process_split(
    source_path: Path,
    output_dir: Path,
    people: Set[str],
    use_symlinks: bool,
    verbose: bool,
    split_name: str
) -> dict:
    """
    Process a single split (train or val).
    
    Args:
        source_path: Path to source images
        output_dir: Path to output directory for this split
        people: Set of person names to include
        use_symlinks: If True, create symlinks instead of copying
        verbose: If True, print progress
        split_name: Name of the split (for logging)
        
    Returns:
        Dictionary with counts of people and images processed
    """
    people_count = 0
    image_count = 0
    
    for person_name in sorted(people):
        person_source = source_path / person_name
        
        if not person_source.exists():
            if verbose:
                print(f"Warning: Person folder not found: {person_name}")
            continue
        
        # Create person folder in output
        person_output = output_dir / person_name
        person_output.mkdir(exist_ok=True)
        
        # Copy or symlink images
        images = list(person_source.glob("*.jpg"))
        
        if not images:
            if verbose:
                print(f"Warning: No images found for {person_name}")
            continue
        
        for img_path in images:
            output_img = person_output / img_path.name
            
            if use_symlinks:
                # Create symbolic link
                if output_img.exists():
                    output_img.unlink()
                output_img.symlink_to(img_path.resolve())
            else:
                # Copy file
                shutil.copy2(img_path, output_img)
            
            image_count += 1
        
        people_count += 1
        
        if verbose and people_count % 100 == 0:
            print(f"  [{split_name}] Processed {people_count} people...")
    
    return {"people": people_count, "images": image_count}


def main():
    parser = argparse.ArgumentParser(
        description="Transform LFW dataset to ImageFolder format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using CSV files for train/val split:
    python transform_data.py \\
        --source data/lfw-deepfunneled/lfw-deepfunneled \\
        --output data \\
        --train-csv data/peopleDevTrain.csv \\
        --val-csv data/peopleDevTest.csv
    
    # Using symlinks to save disk space:
    python transform_data.py \\
        --source data/lfw-deepfunneled/lfw-deepfunneled \\
        --output data \\
        --train-csv data/peopleDevTrain.csv \\
        --val-csv data/peopleDevTest.csv \\
        --symlinks
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to source images directory (lfw-deepfunneled/lfw-deepfunneled)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory (will create train/ and val/ subdirectories)"
    )
    
    parser.add_argument(
        "--train-csv",
        type=str,
        help="Path to peopleDevTrain.csv (training set definition)"
    )
    
    parser.add_argument(
        "--val-csv",
        type=str,
        help="Path to peopleDevTest.csv (validation set definition)"
    )
    
    parser.add_argument(
        "--symlinks",
        action="store_true",
        help="Create symbolic links instead of copying files (saves disk space)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    transform_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        use_symlinks=args.symlinks,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
