#!/usr/bin/env python3
"""
Cleanup Script - Nuke all MLOps resources
Deletes SageMaker endpoints, models, S3 buckets, and CloudFormation stack
"""

import boto3
import argparse
import time
import os

def get_config():
    return {
        'project_name': os.environ.get('PROJECT_NAME', 'mlops-medical-imaging'),
        'region': os.environ.get('AWS_REGION', 'us-east-1'),
        'stack_name': os.environ.get('STACK_NAME', 'mlops-medical-imaging-stack'),
    }

def delete_sagemaker_endpoint(config):
    """Delete SageMaker endpoint, endpoint config, and model"""
    sm_client = boto3.client('sagemaker', region_name=config['region'])
    endpoint_name = f"{config['project_name']}-endpoint"
    
    # Delete endpoint
    try:
        print(f"Deleting endpoint: {endpoint_name}")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        
        # Wait for deletion
        print("Waiting for endpoint deletion...")
        waiter = sm_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name, WaiterConfig={'Delay': 10, 'MaxAttempts': 60})
        print("Endpoint deleted")
    except sm_client.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            print(f"Endpoint {endpoint_name} not found, skipping...")
        else:
            print(f"Error deleting endpoint: {e}")
    
    # Delete endpoint config
    try:
        print(f"Deleting endpoint config: {endpoint_name}")
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print("Endpoint config deleted")
    except sm_client.exceptions.ClientError as e:
        if 'Could not find' in str(e):
            print(f"Endpoint config not found, skipping...")
        else:
            print(f"Error deleting endpoint config: {e}")
    
    # List and delete models
    try:
        models = sm_client.list_models(NameContains=config['project_name'])['Models']
        for model in models:
            print(f"Deleting model: {model['ModelName']}")
            sm_client.delete_model(ModelName=model['ModelName'])
        print(f"Deleted {len(models)} models")
    except Exception as e:
        print(f"Error deleting models: {e}")

def empty_s3_bucket(bucket_name, region):
    """Empty all objects from S3 bucket"""
    s3 = boto3.resource('s3', region_name=region)
    
    try:
        bucket = s3.Bucket(bucket_name)
        print(f"Emptying bucket: {bucket_name}")
        
        # Delete all objects
        bucket.objects.all().delete()
        
        # Delete all object versions (if versioning enabled)
        bucket.object_versions.all().delete()
        
        print(f"Bucket {bucket_name} emptied")
    except Exception as e:
        if 'NoSuchBucket' in str(e):
            print(f"Bucket {bucket_name} not found, skipping...")
        else:
            print(f"Error emptying bucket {bucket_name}: {e}")

def delete_s3_buckets(config):
    """Delete all project S3 buckets"""
    account_id = boto3.client('sts').get_caller_identity()['Account']
    
    buckets = [
        f"{config['project_name']}-training-{account_id}-{config['region']}",
        f"{config['project_name']}-models-{account_id}-{config['region']}",
        f"{config['project_name']}-inference-{account_id}-{config['region']}",
    ]
    
    s3_client = boto3.client('s3', region_name=config['region'])
    
    for bucket_name in buckets:
        # First empty the bucket
        empty_s3_bucket(bucket_name, config['region'])
        
        # Then delete the bucket
        try:
            print(f"Deleting bucket: {bucket_name}")
            s3_client.delete_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} deleted")
        except Exception as e:
            if 'NoSuchBucket' in str(e):
                print(f"Bucket {bucket_name} not found, skipping...")
            else:
                print(f"Error deleting bucket {bucket_name}: {e}")

def delete_cloudformation_stack(config):
    """Delete CloudFormation stack"""
    cf_client = boto3.client('cloudformation', region_name=config['region'])
    
    try:
        print(f"Deleting CloudFormation stack: {config['stack_name']}")
        cf_client.delete_stack(StackName=config['stack_name'])
        
        # Wait for deletion
        print("Waiting for stack deletion...")
        waiter = cf_client.get_waiter('stack_delete_complete')
        waiter.wait(StackName=config['stack_name'], WaiterConfig={'Delay': 10, 'MaxAttempts': 120})
        print("Stack deleted")
    except cf_client.exceptions.ClientError as e:
        if 'does not exist' in str(e):
            print(f"Stack {config['stack_name']} not found, skipping...")
        else:
            print(f"Error deleting stack: {e}")

def delete_training_jobs(config):
    """Stop any running training jobs"""
    sm_client = boto3.client('sagemaker', region_name=config['region'])
    
    try:
        jobs = sm_client.list_training_jobs(
            NameContains=config['project_name'],
            StatusEquals='InProgress'
        )['TrainingJobSummaries']
        
        for job in jobs:
            print(f"Stopping training job: {job['TrainingJobName']}")
            sm_client.stop_training_job(TrainingJobName=job['TrainingJobName'])
        
        print(f"Stopped {len(jobs)} training jobs")
    except Exception as e:
        print(f"Error stopping training jobs: {e}")

def delete_iam_role(config):
    """Delete IAM roles created outside CloudFormation"""
    iam_client = boto3.client('iam')
    
    roles_to_delete = [
        'github-actions-mlops-role',
    ]
    
    for role_name in roles_to_delete:
        try:
            # Detach managed policies
            attached = iam_client.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
            for policy in attached:
                print(f"Detaching policy {policy['PolicyArn']} from {role_name}")
                iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
            
            # Delete inline policies
            inline = iam_client.list_role_policies(RoleName=role_name)['PolicyNames']
            for policy_name in inline:
                print(f"Deleting inline policy {policy_name} from {role_name}")
                iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            
            # Delete role
            print(f"Deleting IAM role: {role_name}")
            iam_client.delete_role(RoleName=role_name)
            print(f"Role {role_name} deleted")
        except iam_client.exceptions.NoSuchEntityException:
            print(f"Role {role_name} not found, skipping...")
        except Exception as e:
            print(f"Error deleting role {role_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Cleanup all MLOps resources')
    parser.add_argument('--confirm', action='store_true', required=True,
                        help='Confirm deletion (required)')
    parser.add_argument('--keep-iam', action='store_true',
                        help='Keep IAM roles')
    parser.add_argument('--keep-buckets', action='store_true',
                        help='Keep S3 buckets')
    args = parser.parse_args()
    
    if not args.confirm:
        print("ERROR: Must pass --confirm to delete resources")
        return
    
    config = get_config()
    print(f"Starting cleanup for project: {config['project_name']}")
    print("=" * 50)
    
    # Order matters - delete dependent resources first
    
    # 1. Stop training jobs
    print("\n[1/6] Stopping training jobs...")
    delete_training_jobs(config)
    
    # 2. Delete SageMaker endpoint
    print("\n[2/6] Deleting SageMaker endpoint...")
    delete_sagemaker_endpoint(config)
    
    # 3. Empty and delete S3 buckets (before CloudFormation)
    if not args.keep_buckets:
        print("\n[3/6] Deleting S3 buckets...")
        delete_s3_buckets(config)
    else:
        print("\n[3/6] Skipping S3 bucket deletion (--keep-buckets)")
    
    # 4. Delete CloudFormation stack
    print("\n[4/6] Deleting CloudFormation stack...")
    delete_cloudformation_stack(config)
    
    # 5. Delete IAM roles
    if not args.keep_iam:
        print("\n[5/6] Deleting IAM roles...")
        delete_iam_role(config)
    else:
        print("\n[5/6] Skipping IAM role deletion (--keep-iam)")
    
    print("\n" + "=" * 50)
    print("Cleanup completed!")

if __name__ == '__main__':
    main()
