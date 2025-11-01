# Data Transformation, Integrity, and Feature Engineering
## 54. Elastic MapReduce (EMR) and Hadoop Overview

- Many new terminologies, such as core node, task node, HDFS, EMRFS, etc.
- Breakdown of EMR
  - Compute - EC2 instance per node
  - Storage - HDFS is fast but not persistent, EMRFS is like HDFS but persists to S3
  - Security - AWS IAM role, KMS key, etc
  - Network - Traffic through usual VPC
  - Database - NA
- Revisit for high level overview of EMR and new terminologies

## 55. EMR Serverless

- AWS CLI only to create and manage EMR serverless in a single region
- Managed capacity
  - Can specify initial worker size & initial capacity
  - Choose a runtime, e.g. Spark, Hive, Presto, etc
  - Manual operations, as it is NOT managed compute
- Breakdown of EMR Serverless
  - Storage - EMRFS persists to S3, same as EMR
  - Security - SSE-S3, SSE-KMS, AWS IAM role, etc
  - Compute - manual lifecycle (unlike EMR on EKS)
    - Created
    - Started
    - Stopped
    - Terminated
- EMR on EKS for both managed compute and managed capacity
- Spark adds 10% overhead memory to compute

## 56. Apache Spark on EMR

- Hadoop is a abstraction that consists of several layers
  - HDFS - Hadoop distributed file system
  - YARN - yet another resource negotiator
  - MapReduce - software framework for big data processing
  - Spark - alternative to MapReduce
    - API for Python, Java, etc
- How Spark works and components
  - Spark SQL
  - Spark Streaming - integrates with Kinesis
  - Spark Core
  - Spark MLLib - distributed ML, e.g. regression, decision trees, K-means, etc
  - GraphX

## 57. Feature Engineering and the Curse of Dimensionality

- Breakdown of feature engineering
  - Attributes of data to create better features to train your model
  - Applied machine learning is basically feature engineering
  - Domain knowledge empowers selecting the most relevant features
- Curse of dimensionality
  - Too many features can lead to sparse data
  - Every feature is a new dimension
  - Unsupervised dimensionality reduction techniques can distill many features into fewer features
    - Principal Component Analysis (PCA)
    - K-Means

## 58. Lab: Preparing Data for TF-IDF with Spark and EMR Studio, Part 1

- Term Frequency and Inverse Document Frequency (TF-IDF)
  - TF measures how often a word occurs in a document
    - Important words for a document
  - DF measures how often a word occurs in an entire set of documents
    - Common words, such as the, a, etc
  - TF / log(IDF) = word score
- Extension of TF-IDF
  - Measure occurences of bi-word or tri-word (bi-gram or tri-gram)
  - Matrix of uni-gram and bi-gram

## 59. Lab: Preparing Data for TF-IDF with Spark and EMR Studio, Part 2

- Search algorithm using TF-IDF
  - Calculate TF-IDF score for every word in a document.
- Navigate to AWS Console > EMR Studio
  - Configure Studio
    - Interactive workload
    - Studio name: TF-IDF
    - Create new bucket
    - Create a service role
    - Workspace name: TF-IDF
    - EMR Serverless
  - Create Studio

## 60. Inputting Missing Data

- Mean replacement
  - Replace missing values with mean value from the rest of the column
  - Use median if column has outliers
  - Generally not very accurate method of imputation
- Drop data
  - Drop a few rows if it doesn't bias your data
  - Almost always not the best method
- Machine Learning
  - KNN: Find K nearest rows and average their values
  - Regression: Find linear or non-linear relations between missing feature and other features
  - Deep Learning: Build a ML to impute data for your ML model, but it's complicated
- Get more data
  - Best method

## 61. Dealing with Unbalanced Data

- Breakdown of unbalanced data
  - Large discrepancy between positive and negative cases
    - For example, fraud cases are rare, and most rows will not be fraud
  - Mainly an issue with neural network
- Oversampling
  - Duplicate samples of minority case randomly
- Undersampling
  - Remove negative rows to balance the data
  - Dropping data is always not the best method
- Synthetic Minority Over-sampling Technique
  - Run KNN of each sample of minority class
  - Duplicate new samples of the minority class using KNN
  - Better than just oversampling
- Adjusting threshold
  - Increase the threshold to reduce false positives, but its cost is more false negatives
  - Decrease the threshold to increase positives, however a risk of more false positives

## 62. Handling Outliers

- Breakdown of outliers
  - Variance is the average of the squared differences from the mean
    - Find the mean of all data
    - For each data, find the squared difference from the mean
    - Find the average of all squared differences (variance)
    - Squared differences are absolute values (ignores negative differences)
  - Standard deviation (SD) is the squared root of variance
  - Data points that lie more than one SD from the mean can be considered an outlier
- Drop outliers
  - Understand why you are doing this
- AWS Random Cut Forest algorithm
  - Found within QuickSight, Kinesis Analytics, SageMaker, etc
  - Outlier detection

## 63. Binning, Transforming, Encoding, Scaling, and Shuffling

- Binning
  - Buckets with a specified range
  - Quantile binning ensures even sizes of bins
  - Transforms numeric data to ordinal data
- Transforming
  - Apply a function to a feature to make it better suited for training
  - Exponential data may benefit from a log transform
- One-hot encoding
  - Transforms categorical data to a numerical representation, e.g. one-hot encoding
  - Very common in deep learning, where categories are represented by individual output neurons.
- Standardizing or normalization
  - Used to ensure that numerical features with different units contribute equally to a model
  - Distribute data normally around a mean of zero
  - Remember to scale your results back up
- Shuffling
  - Remove any bias resulting from the order in which they were collected

## 64. SageMaker is now SageMaker AI

- No video

## 65. SageMaker AI Overview

- Breakdown of SageMaker
  - An AWS service that was introduced before Generative AI
  - Built to handle entire ML workflow, from preparing data to deploying and evaluating models in production
    - Endpoint exposed to client app
    - Model deployed on a server from an ECR image
    - Model trained on a server from an ECR image
    - S3 buckets host both training data and production model artifacts
- Breakdown of SageMaker Notebook
  - Similar to Jupyter notebook
  - Instances on EC2 are spun up from console with S3 data access
  - Runtime models such as Scikit_learn, Spark, Tensorflow
  - Wide variety of built-in models
  - Ability to deploy trained models for making predictions at scale

## 66. SageMaker AI Domains

- Breakdown of SageMaker AI Domain
  - Domain is similar to a workspace, which groups all apps, resources and users together
  - Shared a single EFS volume for all users
  - User profiles own personal applications
    - SageMaker Studio instances
    - Private folder in EFS
  - Shared spaces
    - Public folder in EFS
- By default, a domain has two VPCs
  - Public VPC for Internet access managed by SageMaker AI.
  - Private VPC manually created by user
  - You can change default VPCs

## 67. Data Processing, Training and Deployment with SageMaker AI

- Breakdown of data processing
  - Ingest from S3, Athena, EMR, Redshift, Keyspaces DB
  - Sagemaker integrates with Spark
  - Python packages at your disposal
- Breakdown of training
  - URL of S3 bucket for input/output
  - Choose ML compute resources
  - ECR image of training model
- Breakdown of deployment
  - Persistent endpoint for on-demand user predictions
  - Batch transform to get predictions for an entire dataset
  - Shadow testing evaluates new models against existing models to catch errors
  - Automatic endpoint scaling for HA
  - Sagemaker Neo for deployment to edge devices

## 68. Amazon SageMaker Ground Truth and Label Generation

- Breakdown of SageMaker Ground Truth
  - Labelling training data are outsourced to humans
    - Mechanical Turk workforce
    - Your own internal team
    - Professional labelers
  - Manages a fleet of humans and creates its own model that improves over time
  - As this model learns, only data that the model isn't sure about are sent to human labelers, which can reduce the cost of labelling over time
  - Ground Truth Plus is a turn-key solution
- Other Label Generation
  - AWS Rekognition for images
  - AWS Comprehend for text and documents

## 69. Amazon Mechanical Turk

- Breakdown of Mechanical Turk
  - Crowdsourcing marketplace to perform human tasks
  - Distributed virtual workflows that is relatively cheap
  - You set the reward per task, similar to Fiverr
  - Use cases include data labeler, business processing, etc
  - Integrates with A2I, SageMaker Ground Truth

## 70. SageMaker Data Wrangler

- Limitations
  - Does not have feature for detecting or grouping duplicate records, especially those requiring fuzzy matching.
  - Lacks ML-based features, such as AWS Glue FindMatches.
  - Lacks large scale preprocessing (ETL) features or automating schema inference
- Breakdown of SageMaker Data Wrangler
  - Visual GUI to prepare data for machine learning
  - Ability to import, transform and visualize data
  - Quick model to train your model with your data and measure its results
  - Export as code or notebook, think of it as a code generation tool for your pipeline
- Demo of SageMaker Data Wrangler

## 71. Demo SageMaker Studio, Canvas, and Data Wrangler

- Demo of SageMaker Studio, Canvas and Data Wrangler

## 72. SageMaker Model Monitor and SageMaker Clarify

- Data drift vs Model drift
  - Data drift occurs when the distribution of input data changes over time
    - Usually occurs when new data received is different from what it was trained on
  - Model drift happens when the model's underlying parameters or assumptions become outdated
    - For example, a decline in the accuracy of model predictions
    - Retrain model with updated data
- [SageMaker Model Monitor][r02]
  - Limitations: Does not auto-remediate drifts, but notifications through CloudWatch
  - Useful for continuous tracking data drift, model drift, bias drift, feature attribution drift
  - Metrics are emitted to CloudWatch, and notification through CloudWatch
  - Detects data drift
    - Drift in data quality relative to a baseline
    - Drift in model quality same way as data
    - Drift in bias
    - Drift in feature attribution based on Normalized Discounted Cumulative Gain (NDCG) score
  - Detects anomalies and outliers
  - Detects new features
  - Integrates with SageMaker Clarify
- SageMaker Clarify
  - Limitations: Detects data bias, but not data drift
  - [Model explainability][r03]
    - Helps explain model behaviour, which feature contributes the most to prediction
    - SHAP (Shapeley Additive Explanations)
      - SHAP is based on the concept of a Shapley value
      - Local method that assigns each feature an importance value for a particular perdiction
    - Partial Dependence Plot (PDP)
      - Global method that shows the marginal effect features have on the prediction
      - Unlike SHAP, PDP method is computationally less intensive
  - Metrics for data bias
    - [Conditional Demographic Disparity (CDD)][r04]
      - Focuses on measuring differences in prediction rates, not the correctness of predictions
      - Conditioned on a specific feature like income to detect bias that may not be apparent when considering overall outcomes
      - A facet can be either favoured (majority) or disfavoured (minority)
      - A facet disparity exists when the rate at which a disfavoured facet were rejected exceeds the rate at which they are accepted
        - For example, a disparity exists where women comprise 46% of rejected applicants and 32% of accepted applicants, as the rejected rate exceeds the accepted rate.
      - A conditional facet disparity metric that conditions on a specific feature, such as department, which may reveal a different outcome
        - For example, women were shown to have a higher accepted rate than men when conditioned by department
    - Class Imbalance (CI)
    - Difference in Proportions Labels (DPL)
    - Kullback-Leibler Divergence (KL)
    - Jensen-Shannon Divergence (JS)
    - Lp-norm (LP)

## 73. SageMaker Clarify's Partial Dependence Plots (PDPs), Shapley values, and SHAP

- Breakdown of Partial Dependence Plots (PDPs)
  - Shows dependence of predicted target response on a set of input features
  - Plots show how feature values influence the predictions
- Breakdown of Shapley Values
  - Shapley values are the algorithm used to determine the contribution of each feature towards a model's predictions
  - Originated in game theory, adapted to ML
  - Measures the impact of dropping individual features
  - As it gets complicated with lots of features, SageMaker Clarify uses Shapley Additive explanations (SHAP) as an approximation technique
  - For time series prediction, use asymmetric Shapley Values to determine the contribution of input features at each time step toward predictions

## 74. SageMaker Feature Store

- Breakdown of SageMaker Feature Store
  - A feature is basically a column, attribute or property of an object
  - Challenge to keep it organized and share features across models
  - Aggregates and organizes features in stores, and groups
  - Security: KMS< IAM, TLS, etc

## 75. SageMaker Canvas

- Breakdown of SageMaker Canvas
  - No-code auto-ML GUI for business analyst
  - Upload CSV, create joins
  - Classification or regression
  - Auto-cleansing for missing values, outliers, duplicate, etc.
  - Share models with SageMaker Studio
  - Support for generative AI support via Bedrock, etc

## 76. AWS Glue

- Breakdown of AWS Glue
  - [AWS Glue FindMatches][r01]
    - Feature to detect duplicate records, even when the records are not exact matches
    - Uses machine learning to find similarities across columns
    - Minimal coding required due to ML-based approach
    - Works with large datasets in S3
  - Serverless discovery and definition of table definitions and schema
    - S3 data lake, RDS, Redshift, DB, etc
    - Glue crawler scans data and creates schema
    - How you organized your data will affect how Glue creates its schema
  - Fully managed custom ETL jobs
    - Trigger-driven, scheduled, on-demand
    - Uses Spark under the hood

## 77. AWS Glue Studio

- Breakdown of AWS Glue Studio
  - No-code visual IDE for AWS Glue custom ETL jobs, similar to n8n
  - Connect to many different sources, both AWS and third-party
  - Job dashboard for overview
- Demo of AWS Glue Studio

## 78. AWS Glue Data Quality

- Breakdown of AWS Glue Data Quality
  - Inject a step in your Glue job to auto-evaluate data quality of input sources
  - Data validation for input sources
  - Manual or auto-rules that uses Data Quality Definition Language (DQDL)
  - Results can be used to fail the job, or sent to CloudWatch logs

## 79. AWS Glue DataBrew

- Breakdown of AWS Glue DataBrew
  - A visual IDE for data transformation tool, at $1 per session
  - T tool for ETL, over 250 ready-made Ts
  - Create recipes that can be saved as jobs within a larger project
  - Input from S3, data warehouse, or DB
  - Output to S3
  - Security: KMS, TLS, IAM, CloudWatch and CloudTrail

## 80. Demo: Glue DataBrew

- Demo of AWS Glue DataBrew

## 81. Handling PII in DataBrew Transformations

- Breakdown of Handling Personal Identifiable Information (PII) in DataBrew
  - Subsititution - REPLACE_WITH_RANDOM
  - Shuffling - SHUFFLE_ROWS
  - Deterministic encryption - DETERMINISTIC_ENCRYPT
  - Probabilistic encrpytion - ENCRYPT
  - Decryption - DECRYPT
  - Null out or delete - DELETE
  - Mask out - MASK_CUSTOM, MASK_DATE, etc
  - Hashing - CRYPTOGRAPHIC_HASH

## 82. Intro to Amazon Athena

- Breakdown of Amazon Athena
  - Serverless interactive SQL query engine for S3 data
  - No need to extract data as it stays in S3
  - Supports many formats, e.g. CSV, JSON, Parquet, ORC, Gzip, etc
  - Use cases: query logs, preview data before loading to DB
  - Workgroups to organize users, teams, access, and costs
  - Security: IAM, CloudWatch, ACLs

## 83. Athena and Glue

- Breakdown of Athena and Glue
  - S3 (unstructured) --> Glue (structured) --> Athena (SQL query) --> Quicksight (reports)
  - Athena PAYG cost $5 per TB scanned, failed queries are not charged
    - Save cost by using column formats, such as ORC, Parquet
    - Save cost by using S3 partitions (separate keys)
  - Glue and S3 have their own costs
  - Security: S3 policy, TLS, etc

## 84. Athena and CREATE TABLE AS SELECT

- Breakdown of Athena and CREATE TABLE AS SELECT (CTAS)
  - Creates a table or view from query results
  - Can be used to convert data into a new underlying format, e.g. Parquet
  - Can be used to store data in new S3 object

## 85. Athena Performance

- Breakdown of Athena Performance
  - Use column data, e.g. Parquet, ORC
  - Use S3 partitions
  - A few large files performs better than large number of small files

## 86. Athena ACID Transactions

- Breakdown of Athena ACID Transactions
  - Strict guarantees about transactions
  - Powered by Apache Iceberg under the hood
  - Concurrent users can safely make row-level modifications, use table_type = ICEBERG in your CREATE_TABLE command
  - Compatible with Spark, EMR that supports ICEBERG table format
  - No record locking required
  - Another way of getting ACID is by governed tables from Lake Formation
  - Periodically compact table to improve ACID performance over time

## 87. Athena Fine-Grained Access

- Breakdown of Athena Fine-Grained Access to AWS Glue Data Catalog
  - IAM: DB and table-level security, e.g. glue:GetDatabase
    - Policies to restrict access, such as DROP DATABASE

## Links

[r01]: https://builder.aws.com/content/2c0dOcTyJPnK1NR7H05Oii4HblJ/aws-glue-findmatches-for-incremental-record-matching
[r02]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
[r03]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html
[r04]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-cddl.html