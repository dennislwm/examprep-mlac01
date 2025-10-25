# ML Best Practices
## 277. Intro: Machine Learning Best Practices

## 278. Designing ML Systems with AWS: Responsible AI

- Breakdown of Designing Responsible AI
  - Core Dimensions
    - Fairness
    - Explainability
    - Privacy and Security
    - Safety
    - Controllability
    - Veracity and Robustness
    - Governance
    - Transparency (about Limitations)
- AI Tools for Responsible AI
  - Amazon Bedrock
    - Evaluation tools
  - SageMaker Clarify
    - Bias detection
    - Evaluation
    - Explainability
  - SageMaker Model Monitor
    - Get alerts for inaccurate responses
  - Amazon Augmented AI
    - Humans in the loop to fine tune
  - SageMaker ML Governance
    - SageMaker Role Manager
    - Model Cards (published transparency)
    - Model Dashboard

## 279. ML Design Principles and Lifecycle

- Breakdown ML Design Principles and Lifecycle
  - Assign ownership
  - Provide protection
    - Security controls
  - Enable resiliency
    - Fault tolerance
    - Recoverability
  - Reusability of trained models
  - Reproducibility using version control
  - Optimize resources
  - Reduce cost
  - Enable automation using CI/CD, CT (continuous training)
  - Continuous improvement using monitoring & analysis
  - Minimize environmental impact using managed services
- AWS ML Lifecycle
  - Business Goal -> ML Problem Framing -> Data Processing -> Model Development -> Deployment -> Monitoring
  - Iterative process

## 280. ML Business Goal Identification

- Breakdown of Identifying Business Goal
  - Develop the right skills with accountability and empowerment
  - Discuss and agree on the level of explainability
    - Trade off between model complexity and explainability
    - SageMaker Clarify Experiment feature
        □ Turn off feature to provide explainability
  - Monitor model compliance to business requirements
    - Measure drift and what to monitor
  - Validate data permissions, privacy, software license terms
  - Define overall ROI and opportunity costs
    - Determine KPIs
  - Define the overall environmental impact
    - Use managed services to reduce total cost of ownership (TCO)
    - Tradeoff custom vs pre-trained models

## 281. Framing the ML Problem

- Breakdown Framing the Problem
  - Where to collect data?
  - How to prepare data and which feature is important?
    - Online and offline feature stores
  - Training and tuning model
    - Store in model registry with version control
  - Deploy model
  - Monitor & analysis
    - Feedback loop to data preparation
- What AWS services can you use?
  - SageMaker Role manager
    - Establish user roles and responsibilities
  - SageMaker Experiments
    - AutoML
    - Hyperparameter optimization
  - SageMaker Lineage Tracking
    - Pipelines
    - Studio
    - Feature Store
    - Model Registry
  - SageMaker Model Monitor
    - Define relevant evaluation metrics
    - Establish feedback loops
    - Integrate with CloudWatch
    - Integrate with Amazon Augmented AI for humans in the loop
  - SageMaker Clarify
    - Review fairness and explainability
  - AWS Glue DataBrew
    - Design data encryption and obfuscation
  - AWS API Gateway
    - Abstract change from model consuming applications
  - AWS Lambda, Fargate
    - Adopt a microservice strategy in deployment
  - SageMaker JumpStart, Marketplace
    - Use purpose-built ML services and resources

## 282. Data Processing

- Breakdown of Data Processing
  - Data Collection
    - Label data for supervised learning
    - Ingest data using batch or stream
    - Aggregate data
  - Data Preprocessing
    - Clean
    - Partition
    - Scalable for large data
    - Unbias and balance
    - Augment
  - Feature Engineering
    - Selection
    - Transformation
    - Creation by combining or updating features
    - Extraction
- What AWS services can you use?
  - AWS Data Wrangler, Glue, Athena, Redshift, Quicksight
    - Profile data to improve quality
  - SageMaker Model Registry, SageMaker Experiments
    - Create tracking and version control
  - AWS IAM, KMS, Secrets Manager, VPC, PrivateLink
    - Secure data and model environment
  - AWS Macie
    - Protect sensitive data
  - SageMaker Lineage Tracker
    - Enforce data lineage
  - AWS Comprehend, Transcribe, Athena
    - Keep only relevant data and redact PII
  - AWS Glue
    - Data catalog for schema
  - SageMaker Pipelines
    - Data pipeline and automate CI/CD
  - DataLake
    - Data Architecture
  - Amazon Grouded Truth
    - Managed data labeling
  - Data Wrangler
    - Interactive analysis
  - Feature Store
    - Enable feature reusability

## 283. Model Development

- Breakdown of Model Training & Tuning
  - Feature Selection
  - Building Code
  - Algorithm Selection
  - Training Model using Data/Model Parallel
  - Debugging & Profiling
  - Validation Metrics
  - Hyperparameter Tuning
  - Training Container (contains training code)
  - Model Artifact (result)
- Which AWS services can you use?
  - AWS CloudFormation, CDK, Step Functions, SageMaker Pipelines
    - Automate operations using CI/CD and MLOps
  - AWS ECR, CodeArtifact
    - Establish reliable packaging patterns to access approved public libraries
  - SageMaker internode encryption, EMR encryption in transit
    - Secure internode cluster communications
  - SageMaker Clarify, SageMaker Model Registry & Feature Store rollback
    - Protect against data poisoning threats
  - SageMaker Pipelines
    - Enable CI/CD/CT automation with traceability
    - Establish a model performance evaluation pipeline
  - SageMaker Feature Store
    - Enable feature consistency across training and inference
  - SageMaker Experiments, SageMaker Model Monitor
    - Ensure model validation with relevant data
    - Explore alternatives for performance improvement
    - Archive or delete unnecessary training artifacts
  - SageMaker Clarify
    - Establish data bias detection and mitigation
    - Performance trade-off analysis
        □ Bias vs fairness/variance
        □ Precision vs recall
        □ Accuracy vs complexity
  - Data Wrangler, Model Monitor, Clarify, Experiments
    - Establish feature statistics
  - SageMaker Debugger
    - Detect performance issues when using transfer learning
  - SageMaker Autopilot
    - Use automated machine learning
  - SageMaker, Training Compiler, AWS Managed Spot Instances
    - Use managed training capabilities
    - Tune only the most important hyperparameters
  - SageMaker Distributed Training Libraries
    - Use distributed training
  - SageMaker Lifecycle Configuration, Studio auto-shutdown
    - Stop resources when not in use
  - AWS Budgets, Cost Explorer
    - Setup budget and use resource tagging to track costs
  - SageMaker Debugger, CloudWatch
    - Enable debugging and logging

## 284. Deployment

- Breakdown of Model Deployment
  - Feature Store to fetch features
  - Model Registry to fetch model artifact
  - Container with inference code
  - Endpoint for consumption
- Which AWS services can you use?
  - AWS CloudWatch, EventBridge, SNS
    - Establish deployment environment metrics
  - SageMaker Model Monitor
    - Protect against malicious activities
  - SageMaker Pipelines
    - Automate endpoint changes through a pipeline
  - SageMaker Blue/Green, A/B Testing, linear/canary deployments
    - Use an appropriate deployment strategy
  - SageMaker Neo, IoT Greengrass
    - Edge deployment
    - Cost effective
  - SageMaker Edge
    - Multi-model, multi-container
  - SageMaker Inference Recommender, AutoScaling
    - Right-size the model hosting instance fleet
  - CPU or GPU types
    - Graviton3 for CPU inference
    - Inferentia (inf2) for deep learning inference
    - Trainium (trn1) for training
  - SageMaker Inference Pipelines
    - Deploy multiple models behind a single endpoint

## 285. Monitoring

- Breakdown of Monitoring
- Which AWS service can you use?
  - SageMaker Model Monitor, CloudWatch, Clarify, Model Cards, Lineage Tracking
    - Enable model observability and tracking
    - Evaluate data drift
    - Model explainability
    - Model performance
  - CloudFormation, Model Monitor
    - Synchronize architecture and configuration
  - CloudWatch, GuardDuty, Macie
    - Monitor data for anomalous activities
  - Elastic Inference, AutoScaling
    - Allow automatic scaling of model endpoint
    - Right-size instance fleet
  - SageMaker Pipelines & Projects, CodeCommit, CloudFormation, ECR
    - Ensure a recoverable endpoint with version control
    - Automated retraining
    - Retrain only when necessary
  - SageMaker Data Wrangler
    - Review updated data / features
  - Amazon Augmented AI
    - Humans in the loop feedback
  - Budgets, Tagging
    - Monitor usage and cost
  - Quicksight
    - Return on investment
  - Amazon FSx for Lustre
    - Higher bandwidth for training

## 286. AWS Well-Architected Machine Learning Lens

- Breakdown of AWS Well-Architected ML Lens
  - JSON file that leads you through an interview about how you are using ML
  - https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html
- More Training Resources
  - https://skillbuilder.aws/exam-prep/machine-learning-engineer-associate
  - https://docs.aws.amazon.com/sagemaker/latest/dg
  - https://aws.amazon.com/training/learning-paths/machine-learning/exam-preparation
