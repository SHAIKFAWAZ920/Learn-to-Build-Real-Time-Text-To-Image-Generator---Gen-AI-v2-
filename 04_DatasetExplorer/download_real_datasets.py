import os
import sys
import tarfile
import zipfile
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Dataset Links
OXFORD_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
OXFORD_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100.0, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent:.1f}% ({read_so_far // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
    else:
        sys.stdout.write(f"\rDownloading... {read_so_far // (1024*1024)}MB")
    sys.stdout.flush()

def download_file(url, output_path):
    logger.info(f"Downloading {url} to {output_path}...")
    urllib.request.urlretrieve(url, output_path, reporthook=download_progress)
    print()  # Clear line
    logger.info("Download completed successfully!")

def extract_tgz(filepath, output_dir):
    logger.info(f"Extracting {filepath} to {output_dir}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=output_dir)
    logger.info("Extraction complete!")

def extract_zip(filepath, output_dir):
    logger.info(f"Extracting {filepath} to {output_dir}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    logger.info("Extraction complete!")

def setup_oxford(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "102flowers.tgz")
    labels_path = os.path.join(target_dir, "imagelabels.mat")
    
    # Download images
    if not os.path.exists(archive_path):
        download_file(OXFORD_URL, archive_path)
    else:
        logger.info("Oxford flowers images archive already exists. Skipping download.")
        
    # Download labels
    if not os.path.exists(labels_path):
        download_file(OXFORD_LABELS_URL, labels_path)
    else:
        logger.info("Oxford flowers labels already exists. Skipping download.")
        
    # Extract images
    img_dir = os.path.join(target_dir, "jpg")
    if not os.path.exists(img_dir):
        extract_tgz(archive_path, target_dir)
    else:
        logger.info("Oxford flowers images already extracted. Skipping extraction.")
        
    logger.info(f"Oxford-102 Flowers dataset setup complete at: {target_dir}")

def setup_coco(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    images_zip = os.path.join(target_dir, "val2017.zip")
    annotations_zip = os.path.join(target_dir, "annotations_trainval2017.zip")
    
    # Download images
    if not os.path.exists(images_zip):
        download_file(COCO_VAL_URL, images_zip)
    else:
        logger.info("COCO validation images archive already exists. Skipping download.")
        
    # Download annotations
    if not os.path.exists(annotations_zip):
        download_file(COCO_ANNOTATIONS_URL, annotations_zip)
    else:
        logger.info("COCO annotations archive already exists. Skipping download.")
        
    # Extract images
    val_dir = os.path.join(target_dir, "val2017")
    if not os.path.exists(val_dir):
        extract_zip(images_zip, target_dir)
    else:
        logger.info("COCO validation images already extracted. Skipping extraction.")
        
    # Extract annotations
    ann_dir = os.path.join(target_dir, "annotations")
    if not os.path.exists(ann_dir):
        extract_zip(annotations_zip, target_dir)
    else:
        logger.info("COCO annotations already extracted. Skipping extraction.")
        
    logger.info(f"COCO dataset setup complete at: {target_dir}")

def main():
    print("="*60)
    print("           REAL DATASET DOWNLOAD UTILITY SCRIPT")
    print("="*60)
    print("Select a dataset to download and implement in your drive:")
    print("  [1] Oxford-102 Flowers Dataset (~330MB)")
    print("  [2] COCO Validation 2017 Dataset (~1.2GB)")
    print("  [3] Exit")
    print("-"*60)
    
    choice = input("Enter choice [1-3]: ").strip()
    
    base_dir = os.path.join(os.getcwd(), "dataset")
    
    if choice == "1":
        logger.info("Starting Oxford-102 Flowers dataset implementation...")
        setup_oxford(os.path.join(base_dir, "oxford"))
    elif choice == "2":
        logger.info("Starting COCO Validation 2017 dataset implementation...")
        setup_coco(os.path.join(base_dir, "coco"))
    elif choice == "3":
        logger.info("Exiting download utility.")
    else:
        logger.error("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
