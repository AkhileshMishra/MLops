# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import boto3
import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from monai.config import print_config
from monai.transforms import \
    Compose, LoadImage, Resize, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from monai.networks.nets import densenet121

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def decode_htj2k_dicom(input_path, output_path):
    """
    Decode HTJ2K-encoded DICOM file to standard DICOM format.
    AWS HealthImaging uses HTJ2K (High-Throughput JPEG 2000) compression.
    This function converts it to a format that MONAI/ITK can read.
    """
    try:
        import pydicom
        from pydicom.encaps import decode_data_sequence
        from pydicom.uid import ExplicitVRLittleEndian
        
        # Read the DICOM file
        ds = pydicom.dcmread(input_path)
        
        # Check if it's HTJ2K encoded
        transfer_syntax = str(ds.file_meta.TransferSyntaxUID) if hasattr(ds.file_meta, 'TransferSyntaxUID') else ''
        
        # HTJ2K transfer syntax UIDs
        htj2k_syntaxes = [
            '1.2.840.10008.1.2.4.201',  # HTJ2K Lossless
            '1.2.840.10008.1.2.4.202',  # HTJ2K Lossy
            '1.2.840.10008.1.2.4.203',  # HTJ2K Lossless RPCL
        ]
        
        if transfer_syntax in htj2k_syntaxes:
            logger.info(f"Detected HTJ2K encoded DICOM (Transfer Syntax: {transfer_syntax})")
            
            try:
                # Try to decompress using pydicom's built-in support
                # This requires openjpeg or pillow with JPEG2000 support
                pixel_array = ds.pixel_array
                
                # Create a new dataset with uncompressed data
                ds.PixelData = pixel_array.tobytes()
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                
                # Remove encapsulation-related attributes if present
                if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames == 1:
                    del ds.NumberOfFrames
                
                ds.save_as(output_path)
                logger.info(f"Successfully decoded HTJ2K DICOM to {output_path}")
                return output_path
                
            except Exception as e:
                logger.warning(f"Could not decode HTJ2K with pydicom: {e}")
                # Fall back to using the raw pixel data if available
                try:
                    # Try alternative decoding with imagecodecs
                    import imagecodecs
                    
                    # Get the encapsulated frames
                    frames = decode_data_sequence(ds.PixelData)
                    if frames:
                        # Decode the first frame
                        decoded = imagecodecs.jpeg2k_decode(frames[0])
                        ds.PixelData = decoded.tobytes()
                        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                        ds.save_as(output_path)
                        logger.info(f"Successfully decoded HTJ2K DICOM using imagecodecs to {output_path}")
                        return output_path
                except ImportError:
                    logger.warning("imagecodecs not available for HTJ2K decoding")
                except Exception as e2:
                    logger.warning(f"Alternative HTJ2K decoding failed: {e2}")
        
        # If not HTJ2K or decoding failed, try to convert to a simple format
        # that MONAI can read
        try:
            pixel_array = ds.pixel_array
            
            # Normalize to 0-255 range
            if pixel_array.max() > 255:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Save as PNG which MONAI can definitely read
            png_path = output_path.replace('.dcm', '.png')
            Image.fromarray(pixel_array).save(png_path)
            logger.info(f"Converted DICOM to PNG: {png_path}")
            return png_path
            
        except Exception as e:
            logger.warning(f"Could not extract pixel data: {e}")
            # Return original path and hope for the best
            return input_path
            
    except ImportError as e:
        logger.warning(f"pydicom not available: {e}")
        return input_path
    except Exception as e:
        logger.error(f"Error decoding DICOM: {e}")
        return input_path


class DICOMDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = densenet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=3
    ).to(device) 
    
    print("model_dir is", model_dir)
    print("inside model_dir is", os.listdir(model_dir))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        #model.load_state_dict(torch.load(f)) choose the right way to load your model
        # weights_only=False is required for PyTorch 2.6+ to load models saved with older versions
        model = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    return model.to(device)   


def get_val_data_loader(valX, ValY):
    """Create data loader with transforms that can handle both DICOM and PNG."""
    val_transforms = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        Resize(spatial_size=(512, -1)),
        ToTensor()
    ])
    dataset = DICOMDataset(valX, ValY, val_transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)  # num_workers=0 to avoid multiprocessing issues


NUMPY_CONTENT_TYPE = 'application/json'
JSON_CONTENT_TYPE = 'application/json'

s3_client = boto3.client('s3')


def input_fn(serialized_input_data, content_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Received request of type:{content_type}")
    
    print("serialized_input_data is---", serialized_input_data)
    if content_type == 'application/json':
        
        data = json.loads(serialized_input_data)
        
        bucket = data['bucket']
        image_uri = data['key']
        label = ["normal"]
        
        # Generate unique filename to avoid conflicts
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        original_filename = image_uri.split('/')[-1]
        download_file_name = f'/tmp/{unique_id}_{original_filename}'
        
        print("<<<<download_file_name ", download_file_name)
        print("loaded label is:", label)
        print("bucket:", bucket, " key is: ", image_uri, " download_file_name is: ", download_file_name)
        
        # Download from S3
        s3_client.download_file(bucket, image_uri, download_file_name)
        print('Download finished!')
        
        # Try to decode HTJ2K if needed
        decoded_file = decode_htj2k_dicom(download_file_name, download_file_name)
        print(f'Using file for inference: {decoded_file}')
        
        inputs = [decoded_file]
        val_loader = get_val_data_loader(inputs, label)
        print('get_val_data_loader finished!')

        for i, val_data in enumerate(val_loader):
            # only have one file each iteration
            inputs = val_data[0].permute(0, 3, 2, 1)
            print("input_fn:", inputs)
            
            # Clean up downloaded files
            try:
                if os.path.exists(download_file_name):
                    os.remove(download_file_name)
                if decoded_file != download_file_name and os.path.exists(decoded_file):
                    os.remove(decoded_file)
                print('Removed the downloaded file(s)!')
            except Exception as e:
                print(f'Warning: Could not remove temp files: {e}')
            
            return inputs.to(device)

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    print('Got input Data: {}'.format(input_data))
    print("input_fn in predict:", input_data)
    model.eval()
    response = model(input_data)
    print("response from modeling prediction is", response)
    return response


class_names = ["Normal", "Cap", "Covid"]


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        print("response in output_fn is", prediction_output)
        pred = torch.nn.functional.softmax(torch.tensor(prediction_output), dim=1)
        print("predicted probability: ", pred)
        
        top_p, top_class = torch.topk(pred, 1)
        x = {
            "results": {
                "class": class_names[top_class],
                "probability": round(top_p.item(), 2)
            }
        }
        return json.dumps(x)

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
