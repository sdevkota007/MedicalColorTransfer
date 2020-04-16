import os
import torch

DOWNLOAD_REQUIRED = False
SCAN_TYPES = ["frozenCT", "mri", "normalCT"]
SUPPORTED_IMAGE_FORMATS = ["img", "png"]

# cwd = os.getcwd()
# DATASET_RAW_PATH = os.path.join(cwd, "dataset/lhcftp.nlm.nih.gov/Open-Access-Datasets/Visible-Human/Male-Images/PNG_format/radiological")


DATASET_RAW_PATH = "dataset/lhcftp.nlm.nih.gov/Open-Access-Datasets/Visible-Human/Male-Images/PNG_format/radiological"

USE_CUDA = torch.cuda.is_available()

try:
    from config.local_settings import *
except ImportError as e:
    print(e)
    pass