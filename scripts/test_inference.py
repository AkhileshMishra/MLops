#!/usr/bin/env python3
"""
Test script to validate the inference pipeline
Uploads a test image to S3 and checks for results
"""

import boto3
import json
import time
import os
import argparse

def get_config():
    return {
        'project_name': os.environ.get('PROJECT_NAME', 'mlops-medical-imaging'),
        'region': os.environ.get('AWS_REGION', 'us-east-1'),
    }

def test_endpoint_direct(config):
    """Test SageMaker endpoint directly"""
    sm_runtime = boto3.client('sagemaker-runtime', region_name=config['region'])
    account_id = boto3.client('sts').get_caller_identity()['Account']
    
    endpoint_name = f"{config['project_name']}-endpoint"
    inference_bucket = f"{config['project_name']}-inference-{account_id}-{config['region']}"
    
    # Use a test image from training data
    test_payload = {
        'bucket': f"{config['project_name']}-training-{account_id}-{config['region']}",
        'key': 'training-data/c4da537c-1651-ddae-4486-7db30d67b366-IM0089.dcm'
    }
    
    print(f"Testing endpoint: {endpoint_name}")
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        print(f"\nEndpoint Response: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        return False

def test_s3_trigger(config, test_image_path=None):
    """Test the full S3 trigger pipeline"""
    s3_client = boto3.client('s3', region_name=config['region'])
    account_id = boto3.client('sts').get_caller_identity()['Account']
    
    inference_bucket = f"{config['project_name']}-inference-{account_id}-{config['region']}"
    training_bucket = f"{config['project_name']}-training-{account_id}-{config['region']}"
    
    # Use provided image or copy from training data
    if test_image_path and os.path.exists(test_image_path):
        test_file = test_image_path
        test_key = f"input/{os.path.basename(test_image_path)}"
        
        print(f"Uploading test image: {test_file} -> s3://{inference_bucket}/{test_key}")
        s3_client.upload_file(test_file, inference_bucket, test_key)
    else:
        # Copy from training bucket
        source_key = 'training-data/c4da537c-1651-ddae-4486-7db30d67b366-IM0089.dcm'
        test_key = 'input/test-covid-image.dcm'
        
        print(f"Copying test image from training bucket...")
        print(f"Source: s3://{training_bucket}/{source_key}")
        print(f"Destination: s3://{inference_bucket}/{test_key}")
        
        s3_client.copy_object(
            Bucket=inference_bucket,
            Key=test_key,
            CopySource={'Bucket': training_bucket, 'Key': source_key}
        )
    
    print(f"\nImage uploaded to trigger bucket. Waiting for Lambda processing...")
    
    # Wait for result
    expected_output_key = f"output/{test_key.split('/')[-1].replace('.dcm', '')}_result.json"
    
    max_wait = 120  # 2 minutes
    wait_interval = 5
    elapsed = 0
    
    while elapsed < max_wait:
        try:
            response = s3_client.get_object(Bucket=inference_bucket, Key=expected_output_key)
            result = json.loads(response['Body'].read().decode())
            
            print(f"\n✓ Inference result found!")
            print(f"Output: s3://{inference_bucket}/{expected_output_key}")
            print(f"Result: {json.dumps(result, indent=2)}")
            return True
        except s3_client.exceptions.NoSuchKey:
            print(f"Waiting for result... ({elapsed}s)")
            time.sleep(wait_interval)
            elapsed += wait_interval
    
    print(f"\n✗ Timeout waiting for inference result")
    return False

def check_resources(config):
    """Check if all required resources exist"""
    account_id = boto3.client('sts').get_caller_identity()['Account']
    
    print("Checking resources...")
    
    # Check S3 buckets
    s3_client = boto3.client('s3', region_name=config['region'])
    buckets = [
        f"{config['project_name']}-training-{account_id}-{config['region']}",
        f"{config['project_name']}-models-{account_id}-{config['region']}",
        f"{config['project_name']}-inference-{account_id}-{config['region']}",
    ]
    
    for bucket in buckets:
        try:
            s3_client.head_bucket(Bucket=bucket)
            print(f"  ✓ S3 bucket: {bucket}")
        except:
            print(f"  ✗ S3 bucket: {bucket} (NOT FOUND)")
    
    # Check SageMaker endpoint
    sm_client = boto3.client('sagemaker', region_name=config['region'])
    endpoint_name = f"{config['project_name']}-endpoint"
    
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"  ✓ SageMaker endpoint: {endpoint_name} ({status})")
    except:
        print(f"  ✗ SageMaker endpoint: {endpoint_name} (NOT FOUND)")
    
    # Check Lambda
    lambda_client = boto3.client('lambda', region_name=config['region'])
    lambda_name = f"{config['project_name']}-inference-trigger"
    
    try:
        lambda_client.get_function(FunctionName=lambda_name)
        print(f"  ✓ Lambda function: {lambda_name}")
    except:
        print(f"  ✗ Lambda function: {lambda_name} (NOT FOUND)")

def main():
    parser = argparse.ArgumentParser(description='Test MLOps inference pipeline')
    parser.add_argument('--check', action='store_true', help='Check resources only')
    parser.add_argument('--direct', action='store_true', help='Test endpoint directly')
    parser.add_argument('--trigger', action='store_true', help='Test S3 trigger pipeline')
    parser.add_argument('--image', type=str, help='Path to DICOM image for testing')
    args = parser.parse_args()
    
    config = get_config()
    
    if args.check:
        check_resources(config)
    elif args.direct:
        test_endpoint_direct(config)
    elif args.trigger:
        test_s3_trigger(config, args.image)
    else:
        # Run all tests
        print("=" * 50)
        print("Resource Check")
        print("=" * 50)
        check_resources(config)
        
        print("\n" + "=" * 50)
        print("Direct Endpoint Test")
        print("=" * 50)
        if test_endpoint_direct(config):
            print("\n" + "=" * 50)
            print("S3 Trigger Test")
            print("=" * 50)
            test_s3_trigger(config, args.image)

if __name__ == '__main__':
    main()
