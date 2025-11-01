# Machine Learning Operations (MLOps) with AWS
## 193. Intro: MLOps

## 194. Deployment Guardrails and Shadow Tests

- Breakdown of Deployment Guardrails
  - Blue/Green Deployments
    - All at once: Deploy everything
    - Canary: Deploy a small portion
    - Linear: Deploy linearly
  - Auto-rollbacks
- Shadow Tests
  - Shadow variant: Compare to production
    - Manually promote to production

## 195. SageMaker's Inner Details and Production Variants

- Breakdown of SageMaker
  - All models in SageMaker are hosted in Docker containers
    - Images are saved in a repository, e.g. ECR
  - Pre-built containers: Deep learning, Scikit-learn, Spark ML, Tensorflow, PyTorch, etc
    - Custom training and inference code
  - Supports custom Docker containers - allow for any runtime and language
- Breakdown of SageMaker Docker Structure
  - Structure of Container
    - /opt/ml/
        □ input/
          ® config/
            ◊ hyperparameters.json
            ◊ resourceConfig.json
          ® data/
            ◊ <channel_name>/
              } <input_data>
        □ model/
          ® <inference_files>
        □ code/
          ® <python_files>
        □ output/
          ® <failure>
  - Structure of Docker Image
    - WORKDIR
        □ nginx.conf - Nginx configuration file
        □ predictor.py - Flask webserver for prediction
        □ serve/ - Launches gunicorn server
        □ train/ - Training code
        □ wsgi.py - Wrapper to invoke Flask server
  - Structure of Dockerfile
    - FROM tensorflow/tensorflow:latest
    - RUN pip install sagemaker-training
    - COPY train.py /opt/ml/code/train.py
    - ENV SAGEMAKER_PROGRAM train.py  # SageMaker entrypoint
  - SAGEMAKER environment variables
    - SAGEMAKER_PROGRAM
    - SAGEMAKER_TRAINING_MODULE
    - SAGEMAKER_SERVICE_MODULE
    - SM_MODEL_DIR
    - SM_CHANNELS
    - SM_HPS  # hyperparameters
    - SM_USER_ARGS
- Production Variants
  - Run model variants in parallel during Production
    - Weights to distribute traffic to different variants
    - Perform A/B tests to validate performance in production

## 196. SageMaker on the Edge: Neo and IoT Greengrass

- Breakdown of SageMaker Neo
  - Edge devices: ARM, Intel, Nvidia
  - Neo includes both a compiler and a runtime for edge devices
  - Supports Tensorflow, MXNet, PyTorch, ONNX, XGBoost, DarkNet, Keras, etc
- IOT Greengrass
  - Deploy neo-compiled model to actual edge device
  - Inference at the edge with local data
  - Uses Lambda inference applications
- HTTPS endpoint
  - Deploy neo-compiled model to EC2 instances (for testing)
  - Deploy instance type must be same as training instance type

## 197. SageMaker Resource Management: Instance Types and Spot Training

- Breakdown of Instance Types
  - Deep learning will benefit from GPU instances for training
  - Inference is less demanding and can often use CPU for training
- Managed Spot Training
  - Save up to 90% over On-Demand instances
  - Can be interrupted at any time - use checkpoints to S3 for resume
  - Can increase training time and complexity

## 198. SageMaker Resource Management: Automatic Scaling

- Prerequisites
  - Before you can start auto scaling, you must have already created a SageMaker AI model endpoint
  - The flow is to register the model as a scalable target, define the scaling policy and then apply it.
  - The two options for scaling policies are: target tracking and step scaling
- Scaling Policy
  - Target Tracking
    - Choose a CloudWatch metric and target value, e.g. `InvocationsPerInstance: 70`.
    - Application Auto Scaling will scale out (increase capacity) when InvocationsPerInstance exceeds 70.
    - AAS will scale in (decrease capacity) when InvocationsPerInstance drops below 70.
  - Step Scaling
    - Scale your capacity in predefined increments based on CloudWatch alarms
    - If the breach exceeds the first threshold, AAS will apply the first step adjustment
    - If the breach exceeds the second threshold, AAS will apply the second step adjustment, and so on
    - A cooldown period is used to protect against over-scaling due to multiple breaches occurring in rapid succession
- Breakdown of Auto Scaling
  - Define target metrics, min/max capacity, cooldown periods
  - SageMaker auto-manage number of instances in production
  - Auto-distribute instances across AZs
    - Manually setup logical partitions (subnets) in VPCs across AZs

## 199. SageMaker: Deploying Models for Inference

- Breakdown of Deploying Models for Inference
  - SageMaker Serverless
    - No management of infrastructure
    - Cold starts and uneven traffic
  - SageMaker JumpStart
    - Validate custom models and deploy to pre-configured endpoints
    - Deploy pre-trained models to pre-configured endpoints
  - ModelBuilder Python SDK
    - Configure deployment settings from code
  - AWS CloudFormation
    - AWS::SageMaker::Model resources creates a model to host at an endpoint
    - IaC code for repeatable deployment
- Types of Inference
  - Real-time inference: for low latency interactive workloads
  - Asynchronous inference: low latency for large workloads (queued requests)

## 200. SageMaker Serverless Inference and Recommender

- Breakdown of SageMaker Serverless Inference
  - Specify your container memory requirements, concurrency number, etc
  - Managed underlying capacity scaled down to zero if no traffic
  - Monitor via CloudWatch
- Serverless Recommender (not ML Recommender)
  - Auto-tuning and auto-load testing
  - Deploys to optimal inference endpoint
  - Benchmark different endpoint configurations
  - Auto-recommends instance types and configurations

## 201. SageMaker Inference Pipelines

- Breakdown of SageMaker Inference Pipelines
  - Combine a linear sequence of 2-15 Docker containers
    - Perform both real-time and batch inferences
    - Pre-processing, predictions, post-processing
  - Any combination of pre-trained or custom models in Docker containers

## 202. SageMaker Model Monitor

- Breakdown of SageMaker Model Monitor
  - Detect anomalies & outliers
  - Detect new features
    - No code needed
  - Visualized data drift
    - Get alerts on data drift via CloudWatch
    - Manually scheduled monitoring job
        □ Create a baseline data quality
    - Drift: data statistical, model accuracy, bias, feature attribution
        □ Use Normalized Discounted Cumulative Gain (NDCG) for feature attribution drift
  - Data stored in S3
  - Integrates with Clarify, Tensorboard, Quicksight, Ground Truth
- SageMaker Clarify
  - Detects potential bias
    - Unbalanced columnar data
  - Explains model behaviour
    - Understand which feature contributes the most to predictions

## 203. Model Monitor Data Capture

- Breakdown of Model Monitor Data Capture
  - Logs inputs from your endpoint to S3 as JSON
  - Logs outputs from your endpoint
  - Supports both real-time and batch outputs
  - Allows encryption of logs

## 204. MLOps with SageMaker, Kubernetes, SageMaker Projects, and Pipelines

- Breakdown with MLOps with SageMaker and Kubernetes
  - SageMaker Operators for Kubernetes
  - SageMaker Components for Kubeflow Pipelines
    - Processing
    - Hyperparameter Tuning
    - Training
    - Inference
  - Enables hybrid ML workflows (on-prem + cloud)
- SageMaker Projects
  - Native MLOps with CI/CD
    - Build images
    - Prep data
    - Train models
    - Evaluate models
    - Deploy models
    - Monitor models
  - Use code repository
  - Use SageMaker pipelines
- SageMaker Pipelines
  - Useful for orchestrating ML workflows, visualize them as a DAG, and leverages ML Lineage Tracking for audit and compliance
  - Conditional steps to introduce manual approvals in the ML workflow
  - Integrations
    - Seamless integration with ML Lineage Tracking
    - SageMaker Pipelines [callback step][r01] waits for an external task, such as AWS Glue job, to complete. The output of the task, stored in S3, is then passed to subsequent steps in the pipeline.
  - Limitations
    - AWS Glue job cannot be directly integrated as processing step, as it is not a SageMaker-native processing task
- [SageMaker ML Lineage Tracking][r03]
  - Limitations
    - Does not enforce governance
    - Lacks built-in mechanisms for validating or approving models for deployment
  - Creates and stores information about steps of a ML workflow from data preparation to model deployment
    - Automatically creates a connected graph of lineage entity metadata tracking your workflow
    - Input artifacts -> Processing component -> Dataset artifact, Image artifact, etc -> Training Job component -> Model artifact -> Model Package artifact -> Endpoint action -> Endpoint context
  - Reproduce workflow steps, track model and dataset lineage, and establish model governance and audit standards
  - Keep a running history of model discovery experiments


## 205. What is Docker?

## 206. Amazon ECS

- Breakdown of Elastic Container Service (ECS)
  - Launch an ECS task on an ECS cluster when deploying a Docker container
  - Launch type: EC2 instance
    - Manually provision and register EC2 instance
    - Run ECS agent to register an EC2 instance in the ECS cluster
    - AWS will manage Docker containers within your EC2 instances
  - Launch type: Fargate
    - No EC2 instances to manage
    - Manually create Task definitions
    - AWS will manage ECS tasks for you based on CPU/RAM you need
  - Security: IAM roles
    - EC2 Instance Profiles for EC2 instance
        □ Used by ECS agent
    - EC2 Task role for Fargate
        □ Used by Task definitions
  - Integrates with Load Balancer
    - ALB works for most use cases
    - NLB only for high throughput
    - Classic not recommended
  - Integrates with EFS
    - Mount EFS onto EC2 or Fargate Tasks
    - EFS serverless + Fargate serverless
        □ S3 cannot be mounted onto Fargate Tasks

## 207. Amazon ECS - Create Cluster - Hands On

## 208. Amazon ECS - Create Service - Hands On

## 209. Amazon ECR

- Breakdown of Elastic Container Registry (ECR)
  - Similar to DockerHub
  - Private and public registry
  - Integrated with ECS, backed by S3
  - Supports image lifecycle, tagging, and vulnerability scanning

## 210. Amazon EKS

- Breakdown of Elastic Kubernetes Service (EKS)
  - Fully managed Kubernetes, similar to ECS
  - Launch types
    - Supports both EC2 launch type and Fargate launch type
  - Node types
    - Managed nodes are part of an ASG managed and registered by EKS
    - Self-managed nodes managed and registered by you
    - Fargate nodes - no server
  - Cloud agnostic between Azure, Google, etc
  - Storage
    - Need to specify StorageClass manifest on EKS cluster
    - Leverage Container Storage Interface (CSI) compliant driver
    - Support EBS
    - Support EFS (works with Fargate)

## 211. Amazon EKS - Hands On

## 212. AWS Batch

- Breakdown of Batch
  - Run serverless batch jobs
    - Must provide a Docker image
  - Fully managed provisioning of instances (EC2 & Spot)
  - Schedule batch jobs using CloudWatch
  - Orchestrate batch jobs using Step Functions
- Batch vs Glue
  - Glue for ETL
  - Batch for any computing task

## 213. AWS CloudFormation

- Infrastructure Composer
  - Visualize Infrastructure as code as a diagram
    - Shows all resources and relationships

## 214. AWS CloudFormation - Hands On

## 215. AWS CDK

- SAM
  - Serverless focus
    - Great for Lambda
  - Template declaration in JSON or YAML
    - Leverage CloudFormation
- CDK
  - All AWS resources
  - Uses a programming language, such as Python, etc
- SAM + CDK
  - Use SAM CLI to locally test your CDK application
    - Must run cdk synth

## 216. AWS CDK - Hands On

## 217. AWS CodeDeploy

- Breakdown of CodeDeploy
  - Auto-deploy application on EC2 instances, and on-prem servers
    - Manually install CodeDeploy agent
  - Advanced AWS service

## 218. AWS CodeBuild

- Breakdown of CodeBuild
  - Compiles, test, and build packages that are available to be deployed (by CodeDeploy).
  - Fully managed and serverless
  - Use CodePipeline to orchestrate steps
    - CodeCommit -> CodeBuild -> CodeDeploy

## 219. AWS CodePipeline

- Breakdown of CodePipeline
  - Orchestrate steps to have code pushed to production
  - Code -> Build -> Test -> Provision -> Deploy
  - Fully managed CI/CD
  - Integrates with CodeCommit, CodeBuild, CodeDepl;oy, CloudFormation, GitHub, etc

## 220. Git Review: Architecture and Commands

## 221. Gitflow, GitHub Flow

- Gitflow
  - Feature, Dev, Release, Hotfix, Main branches
- GitHub Flow
  - Only two branches: Main and feature

## 222. Amazon EventBridge

- Breakdown of EventBridge
  - Formerly known as CloudWatch Events
    - Schedule cron jobs
    - React to events, e.g. IAM User login
- EventBridge Rules
  - Rule type
    - Scheduled
    - React to a combination of Events
  - Trigger a Target with a JSON payload
    - Lambda, AWS Batch, ECS Task, SQS, SNS, Kinesis, etc
    - External API
- EventBridge Pipes
  - Connects an event source to a target with optional filtering
- Event Buses
  - Third-party events sent to Partner Event Bus
  - AWS events sent to Default Event Bus
  - Custom application events sent to Custom Event Bus
  - Archive events for replaying
- Schema Registry
  - Managed schemas for each AWS resource
    - Download code bindings for your application
  - Schemas can be versioned
- Resource-based policy
  - Managed permissions for each event bus

## 223. Amazon EventBridge - Hands On

## 224. AWS Step Functions

- Breakdown of Step Functions
  - Manually design and visualize workflows
  - Auto error-handling and retry outside of code
  - Audit history of workflow runs
  - Allow wait for arbitrary amount of time between steps
  - Max execution time of a state machine is one year
  - Amazon State Language (JSON)

## 225. AWS Step Functions: State Machines and States

- Breakdown of State Machines and States
  - State Machine defines your Step Function workflow
    - State defines each step
  - Types of states
    - Task - Action
    - Choice - Conditional logic
    - Wait - Delay
    - Parallel - Adds separate branches for execution
    - Map - Run a subset of steps, in parallel, for each item in dataset
        □ Specifically used in data engineering
        □ Works with JSON, S3 objects, CSV files
    - Pass - Move to next step
    - Succeed or Fail

## 226. Amazon Managed Workflows for Apache Airflow (MWAA)

- Breakdown of MWAA
  - Apache Airflow is a batch-oriented workflow
  - Develop, schedule and monitor your workflow
  - Workflows are defined as Python code
    - Creates a Directed Acyclic Graph (DAG)
    - Visualize a DAG
  - Managed State Machine written as code with Apache Airflow
  - Integrates with AWS services, e.g. S3, Redshift, etc

## 227. AWS Lake Formation

- [AWS Lake Formation][r02]
  - Specifically designed to aggregate and centrally managed large datasets from S3, databases, etc
  - Large scale preprocessing
  - Built on top of AWS Glue
    - No cost for Lake Formation
    - Cost for underlying services, such as Glue, S3, Redshift, etc
  - Makes it easy to setup a Data Lake
    - Loading data from external source, e.g. S3, RDS, etc
    - Auto setup partitions
    - Encryption and managed keys
    - Defined ETL jobs
    - Monitoring jobs
    - Auditing jobs
  - Manual steps to Lake Formation
    - Create IAM user for Data Analyst
    - Create AWS Glue connection to your data sources
    - Create S3 bucket for Data Lake
    - Register the S3 path in Lake Formation
    - Create RDS in Lake Formation for data catalog
    - Use a blueprint for a workflow
    - Run the workflow
    - Grant SELECT permissions for users from Redshift, Athena, EMR, etc
  - Integrates with Redshift, Athena, EMR, etc
  - Supports ACID transactions with Governed Tables
    - Cannot change table type afterwards
    - Auto compaction for storage optimization
    - Granular access control with row and cell level security

## 228. Lake Formation Data Filters

- Breakdown of Data Filters
  - Applied when granting SELECT permissions on tables
  - Granular access control with row and cell level security

## Links

[r01]: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html#step-type-callback
[r02]: https://docs.aws.amazon.com/lake-formation/latest/dg/what-is-lake-formation.html
[r03]: https://docs.aws.amazon.com/sagemaker/latest/dg/lineage-tracking.html