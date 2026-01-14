# MLOps Pipeline Setup Guide

This guide walks you through deploying the Medical Imaging MLOps pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Training Pipeline (GitHub Actions - Manual Trigger)        │
├─────────────────────────────────────────────────────────────────────┤
│ GitHub Actions → SageMaker Training Job → Model → Deploy Endpoint   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Inference Pipeline (S3 Image Drop Trigger)                 │
├─────────────────────────────────────────────────────────────────────┤
│ S3 (input/*.dcm) → Lambda → SageMaker Endpoint → S3 (output/*.json) │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. AWS Account with appropriate permissions
2. GitHub repository with Actions enabled
3. AWS IAM OIDC provider configured for GitHub Actions

## Setup Steps

### Step 1: Configure GitHub Secret

Add the following secret to your GitHub repository:
- Go to: Settings → Secrets and variables → Actions
- Create secret: `AWS_ROLE_ARN`
- Value: `arn:aws:iam::637423443220:role/github-actions-mlops-role`

### Step 2: Deploy Infrastructure

1. Push to main branch (or manually trigger)
2. Go to: Actions → "Deploy Infrastructure" → Run workflow
3. Wait for completion (~5 minutes)

This creates:
- S3 buckets (training, models, inference)
- IAM roles for SageMaker and Lambda
- Lambda function for inference trigger
- S3 event notification

### Step 3: Train and Deploy Model

1. Go to: Actions → "Train and Deploy Model" → Run workflow
2. Configure options:
   - `epochs`: Number of training epochs (default: 50)
   - `skip_deploy`: Set to false to deploy endpoint
3. Wait for completion (~30-45 minutes)

This:
- Uploads training data to S3
- Runs SageMaker training job (ml.p3.2xlarge)
- Deploys real-time inference endpoint (ml.m5.large)

### Step 4: Test the Pipeline

#### Option A: Upload via AWS Console
1. Go to S3 console
2. Navigate to: `mlops-medical-imaging-inference-{account_id}-us-east-1`
3. Upload a DICOM file to the `input/` folder
4. Check `output/` folder for results (within 30 seconds)

#### Option B: Upload via CLI
```bash
# Upload test image
aws s3 cp your-image.dcm s3://mlops-medical-imaging-inference-{account_id}-us-east-1/input/

# Check results
aws s3 ls s3://mlops-medical-imaging-inference-{account_id}-us-east-1/output/

# Download result
aws s3 cp s3://mlops-medical-imaging-inference-{account_id}-us-east-1/output/your-image_result.json .
```

#### Option C: Use test script
```bash
cd scripts
pip install boto3
python test_inference.py --trigger
```

## Result Format

```json
{
  "input_file": "s3://bucket/input/image.dcm",
  "timestamp": "2024-01-14T12:00:00.000000",
  "prediction": {
    "results": {
      "class": "Covid",
      "probability": 0.95
    }
  }
}
```

Classes:
- `Normal` - No infection detected
- `Cap` - Community Acquired Pneumonia
- `Covid` - COVID-19 infection

## Cleanup (Nuke All Resources)

1. Go to: Actions → "Cleanup All Resources" → Run workflow
2. Type `DELETE` in the confirmation field
3. Optionally keep IAM roles or S3 buckets
4. Wait for completion (~10 minutes)

## Cost Estimates

| Resource | Cost |
|----------|------|
| Training (ml.p3.2xlarge, ~30 min) | ~$2.00 per run |
| Inference Endpoint (ml.m5.large) | ~$0.13/hour |
| Lambda | Free tier (likely) |
| S3 | Minimal |

**Tip**: Delete the endpoint when not in use to save costs:
```bash
aws sagemaker delete-endpoint --endpoint-name mlops-medical-imaging-endpoint
```

## Troubleshooting

### Endpoint not responding
```bash
aws sagemaker describe-endpoint --endpoint-name mlops-medical-imaging-endpoint
```

### Lambda not triggering
Check CloudWatch Logs:
```bash
aws logs tail /aws/lambda/mlops-medical-imaging-inference-trigger --follow
```

### Training job failed
```bash
aws sagemaker describe-training-job --training-job-name <job-name>
```

## File Structure

```
├── .github/workflows/
│   ├── deploy-infrastructure.yaml  # Deploy/destroy infrastructure
│   ├── train-and-deploy.yaml       # Train model and deploy endpoint
│   └── cleanup.yaml                # Nuke all resources
├── infrastructure/
│   └── mlops-infrastructure.yaml   # CloudFormation template
├── scripts/
│   ├── train_pipeline.py           # Training orchestration
│   ├── cleanup.py                  # Resource cleanup
│   └── test_inference.py           # Test utilities
└── Classification/
    ├── code/
    │   ├── train.py                # Training script
    │   ├── inference.py            # Inference handlers
    │   └── requirements.txt        # Dependencies
    └── data/                       # Sample DICOM images
```
