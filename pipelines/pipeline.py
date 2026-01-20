"""
SageMaker Pipeline for ICU Requirement Prediction

This module orchestrates the ML workflow using SageMaker Pipelines:
1. Data preparation step
2. Training step
3. Evaluation step
4. Model registration (optional)
"""

import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

# Configuration
S3_BUCKET = "icu-required"
S3_KEY = "cleaned_data.csv"
AWS_REGION = "us-east-1"
PIPELINE_NAME = "ICU-Requirement-Prediction-Pipeline"

# SageMaker configuration
INSTANCE_TYPE_PROCESSING = "ml.m5.large"
INSTANCE_TYPE_TRAINING = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"  # XGBoost framework version
SKLEARN_VERSION = "1.2-1"


def get_pipeline(
    role: str,
    s3_bucket: str = S3_BUCKET,
    s3_prefix: str = "icu-prediction",
    region: str = AWS_REGION
) -> Pipeline:
    """
    Create and return the SageMaker Pipeline.
    
    Args:
        role: SageMaker execution role ARN
        s3_bucket: S3 bucket for artifacts
        s3_prefix: S3 prefix for pipeline artifacts
        region: AWS region
        
    Returns:
        Pipeline object
    """
    
    # Initialize SageMaker session
    pipeline_session = PipelineSession()
    
    # Pipeline parameters
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{s3_bucket}/{S3_KEY}"
    )
    
    n_splits = ParameterInteger(name="NFolds", default_value=5)
    n_repeats = ParameterInteger(name="NRepeats", default_value=10)
    
    # =========================================================================
    # Step 1: Data Preparation
    # =========================================================================
    sklearn_processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        instance_type=INSTANCE_TYPE_PROCESSING,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )
    
    data_prep_step = ProcessingStep(
        name="DataPreparation",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{s3_bucket}/{s3_prefix}/processed"
            )
        ],
        code="src/data_prep.py",
    )
    
    # =========================================================================
    # Step 2: Training with XGBoost
    # =========================================================================
    # Note: For full cross-validation, we use a processing step instead
    # of native XGBoost training step to maintain our custom CV logic
    
    training_processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        instance_type=INSTANCE_TYPE_TRAINING,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )
    
    training_step = ProcessingStep(
        name="ModelTraining",
        processor=training_processor,
        inputs=[
            ProcessingInput(
                source=data_prep_step.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output",
                destination=f"s3://{s3_bucket}/{s3_prefix}/model"
            )
        ],
        code="src/train.py",
    )
    
    # =========================================================================
    # Step 3: Evaluation
    # =========================================================================
    evaluation_processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        instance_type=INSTANCE_TYPE_PROCESSING,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )
    
    evaluation_step = ProcessingStep(
        name="ModelEvaluation",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ProcessingOutputConfig.Outputs[
                    "model"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=data_prep_step.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                destination=f"s3://{s3_bucket}/{s3_prefix}/evaluation"
            )
        ],
        code="src/evaluate.py",
    )
    
    # =========================================================================
    # Create Pipeline
    # =========================================================================
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data, n_splits, n_repeats],
        steps=[data_prep_step, training_step, evaluation_step],
        sagemaker_session=pipeline_session,
    )
    
    return pipeline


def run_pipeline_local():
    """
    Run the pipeline locally (without SageMaker infrastructure).
    This is useful for development and testing.
    """
    import subprocess
    import sys
    
    print("="*60)
    print("RUNNING PIPELINE LOCALLY")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Data Preparation
    print("\n[Step 1/3] Data Preparation...")
    result = subprocess.run(
        [sys.executable, "src/data_prep.py"],
        cwd=base_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error in data prep: {result.stderr}")
        return False
    print(result.stdout)
    
    # Step 2: Training
    print("\n[Step 2/3] Model Training...")
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        cwd=base_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error in training: {result.stderr}")
        return False
    print(result.stdout)
    
    # Step 3: Evaluation
    print("\n[Step 3/3] Model Evaluation...")
    result = subprocess.run(
        [sys.executable, "src/evaluate.py"],
        cwd=base_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error in evaluation: {result.stderr}")
        return False
    print(result.stdout)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    return True


def main():
    """Main function to create or run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ICU Prediction SageMaker Pipeline")
    parser.add_argument("--local", action="store_true", help="Run pipeline locally")
    parser.add_argument("--create", action="store_true", help="Create/update pipeline in SageMaker")
    parser.add_argument("--execute", action="store_true", help="Execute pipeline in SageMaker")
    parser.add_argument("--role", type=str, help="SageMaker execution role ARN")
    args = parser.parse_args()
    
    if args.local:
        run_pipeline_local()
    elif args.create or args.execute:
        if not args.role:
            print("Error: --role is required for SageMaker operations")
            return
        
        pipeline = get_pipeline(role=args.role)
        
        if args.create:
            print("Creating/updating pipeline...")
            pipeline.upsert(role_arn=args.role)
            print(f"Pipeline '{PIPELINE_NAME}' created/updated successfully.")
        
        if args.execute:
            print("Starting pipeline execution...")
            execution = pipeline.start()
            print(f"Pipeline execution started: {execution.arn}")
    else:
        print("Usage:")
        print("  --local    Run pipeline locally")
        print("  --create   Create/update pipeline in SageMaker (requires --role)")
        print("  --execute  Execute pipeline in SageMaker (requires --role)")


if __name__ == "__main__":
    main()
