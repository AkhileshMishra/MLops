#!/usr/bin/env python3
"""
SageMaker Training Pipeline Script
Trains the COVID CT Classification model and deploys endpoint
"""

import boto3
import sagemaker
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import argparse
import os
import time
import json

def get_config():
    """Get configuration from environment or defaults"""
    return {
        'project_name': os.environ.get('PROJECT_NAME', 'mlops-medical-imaging'),
        'region': os.environ.get('AWS_REGION', 'us-east-1'),
        'role_arn': os.environ.get('SAGEMAKER_ROLE_ARN'),
        'training_instance': os.environ.get('TRAINING_INSTANCE', 'ml.p3.2xlarge'),
        'inference_instance': os.environ.get('INFERENCE_INSTANCE', 'ml.m5.large'),
        'epochs': int(os.environ.get('EPOCHS', '50')),
    }

def upload_training_data(sess, bucket, local_path='Classification/data'):
    """Upload training data to S3"""
    print(f"Uploading training data to s3://{bucket}/training-data/")
    
    s3_path = sess.upload_data(
        path=local_path,
        bucket=bucket,
        key_prefix='training-data'
    )
    print(f"Training data uploaded to: {s3_path}")
    return s3_path

def run_training(config, sess, training_data_uri):
    """Run SageMaker training job"""
    print("Starting SageMaker training job...")
    
    account_id = boto3.client('sts').get_caller_identity()['Account']
    models_bucket = f"{config['project_name']}-models-{account_id}-{config['region']}"
    
    metrics = [
        {"Name": "train:loss", "Regex": "average loss: ([0-9\\.]+)"},
        {"Name": "train:f1", "Regex": "f1 score is:([0-9\\.]+)"},
        {"Name": "train:accuracy", "Regex": "current AUC: ([0-9\\.]+)"},
    ]
    
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='Classification/code',
        role=config['role_arn'],
        framework_version='1.13.1',
        py_version='py39',
        instance_count=1,
        instance_type=config['training_instance'],
        output_path=f"s3://{models_bucket}/training-output",
        hyperparameters={
            'seed': 42,
            'lr': 1e-5,
            'epochs': config['epochs'],
            'batch-size': 4,
        },
        metric_definitions=metrics,
        max_run=3600,  # 1 hour max
        base_job_name=f"{config['project_name']}-training",
    )
    
    estimator.fit({'train': training_data_uri}, wait=True)
    
    print(f"Training completed. Model artifacts: {estimator.model_data}")
    return estimator

def deploy_endpoint(config, estimator):
    """Deploy or update SageMaker endpoint"""
    endpoint_name = f"{config['project_name']}-endpoint"
    
    print(f"Deploying endpoint: {endpoint_name}")
    
    # Check if endpoint exists
    sm_client = boto3.client('sagemaker', region_name=config['region'])
    
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        print(f"Endpoint {endpoint_name} exists, will update...")
    except sm_client.exceptions.ClientError:
        endpoint_exists = False
        print(f"Endpoint {endpoint_name} does not exist, will create...")
    
    # Deploy
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=config['inference_instance'],
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        update_endpoint=endpoint_exists,
    )
    
    print(f"Endpoint deployed: {endpoint_name}")
    return predictor

def save_deployment_info(config, estimator):
    """Save deployment information for reference"""
    account_id = boto3.client('sts').get_caller_identity()['Account']
    
    info = {
        'endpoint_name': f"{config['project_name']}-endpoint",
        'model_data': estimator.model_data,
        'training_job_name': estimator.latest_training_job.name,
        'inference_bucket': f"{config['project_name']}-inference-{account_id}-{config['region']}",
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }
    
    with open('deployment_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Deployment info saved: {json.dumps(info, indent=2)}")
    return info

def main():
    parser = argparse.ArgumentParser(description='Train and deploy medical imaging model')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, only deploy')
    parser.add_argument('--skip-deploy', action='store_true', help='Skip deployment')
    args = parser.parse_args()
    
    config = get_config()
    
    if not config['role_arn']:
        # Try to get from CloudFormation exports
        cf_client = boto3.client('cloudformation', region_name=config['region'])
        try:
            exports = cf_client.list_exports()['Exports']
            for export in exports:
                if export['Name'] == f"{config['project_name']}-sagemaker-role-arn":
                    config['role_arn'] = export['Value']
                    break
        except Exception as e:
            print(f"Could not get role from CloudFormation: {e}")
    
    if not config['role_arn']:
        raise ValueError("SAGEMAKER_ROLE_ARN not set and could not be retrieved from CloudFormation")
    
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize SageMaker session
    sess = sagemaker.Session()
    account_id = boto3.client('sts').get_caller_identity()['Account']
    training_bucket = f"{config['project_name']}-training-{account_id}-{config['region']}"
    
    if not args.skip_training:
        # Upload training data
        training_data_uri = upload_training_data(sess, training_bucket)
        
        # Run training
        estimator = run_training(config, sess, training_data_uri)
        
        if not args.skip_deploy:
            # Deploy endpoint
            deploy_endpoint(config, estimator)
            
            # Save deployment info
            save_deployment_info(config, estimator)
    
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()
