# Management and Governance
## 256. Intro: Management and Governance

## 257. Amazon CloudWatch - Metrics

- Breakdown
  - Metric belongs to a namespace
  - A dimension (up to 30) is an attribute of a metric
  - Metric has a timestamp every 5 mins (up to 1 per min)
  - Custom metric can be created by user, e.g. memory usage
  - Stream of metric to Kinesis Data Firehose
    - S3, Redshift, OpenSearch
    - Datadog, etc
  - Example metric:
    - EC2 instance: CPU utilization

## 258. Amazon CloudWatch - Logs

- Breakdown
  - Define a log group, representing an application
  - Log stream is an instance within an application
  - Log expiration policy: never expire, 1 day to 10 years
  - Logs aggregation from multiple accounts and regions
  - Send logs to
    - S3 (batch export up to 12 hours)
    - Kinesis Data Streams (subscription real-time logs)
    - Kinesis Data Firehose
    - AWS Lambda
  - Sources of logs
    - SDK
    - CloudWatch Logs Agent
    - Elastic Beanstalk, etc
  - Custom metric based on string pattern found in logs
- CloudWatch Logs Insights
  - Visual log query dashboard
  - Log query engine
  - Save queries
  - Query multiple log groups

## 259. Amazon CloudWatch - Logs - Hands On

## 260. Amazon CloudWatch - Logs Unified Agent

- Breakdown
  - By default, EC2 logs will not go to CloudWatch
  - Manually run CloudWatch agent
  - EC2 IAM profile must have IAM permissions to write to CloudWatch
  - CloudWatch Unified Agent > CloudWatch Log Agent
- CloudWatch Logs Agent
  - Can send to CloudWatch logs
- CloudWatch Unified Agent
  - Collect granular EC2 metrics such as memory, etc
    - CPU: active, guest, idle, system, user, etc
    - Disk: free, used, total
    - Memory: free, inactive
    - Netstat
    - Processes
    - Swap space
  - Can centralized configuration using SSM Parameter Store
  - Can send to CloudWatch logs

## 261. Amazon CloudWatch - Alarms

- Breakdown of CloudWatch - Alarms
  - Trigger notification for any metric
  - Three states: OK, INSUFFICIENT_DATA, ALARM
  - Period in seconds to evaluate the metric
  - Targets:
    - EC2:
        □ Instance status: STOPPED, etc
        □ System status: FAILED, etc
        □ EBS status
    - Trigger EC2 instance recovery
    - Trigger Auto scaling action
    - Send notification to SNS
  - Test alarm using AWS cloudwatch set-alarm-state
- Composite Alarms
  - Monitor states of multiple alarms: AND, OR conditions
  - Helpful to reduce false positives

## 262. Amazon CloudWatch - Alarms - Hands On

## 263. AWS X-Ray

- Breakdown of AWS X-Ray
  - AWS X-Ray > Application CloudWatch logs
  - Visual trace of applications
    - Follow a client request end-to-end
    - Each component adds its own trace
    - Trace is made of segments
    - Annotations added at each component
    - Trace every request or a sample of requests
  - Security using IAM, KMS etc
  - Advantages of X-Ray
    - Troubleshoot performance for SLA
    - Understand dependencies
    - Pinpoint issues
    - Find errors and exceptions
  - Integrates with Lambda, Beanstalk, ECS, API Gateway, EC2
- X-Ray Support
  - Import AWS SDK in Java, Python, Go, Node.js, .NET
  - Install X-Ray agent on EC2 (IAM profile has proper permissions)
  - Manually enable X-Ray on Lambda
    - Lambda must have proper permissions
    - Enable X-Ray Active Tracing
  - Application must have IAM write permission to X-Ray

## 264. AWS X-Ray - Hands On

## 265. Overview of Amazon Quicksight

- Breakdown of Quicksight
  - Business analytics serverless service to perform interactive ad-hoc analysis
  - Data sources: Redshift, RDS, Athena (S3 Data Lake), EC2-hosted database, Excel (S3)
  - AWS Glue > Limited ETL
  - Data sets are imported into SPICE (Superfast, Parallel, In-memory, Calculation Engine)
  - Security: VPC (IP address), row-level or col-level authorization, User Management
- Machine Learning Insights
  - Anomaly detection
  - Auto-narratives
  - Forecasting
- Quicksight Q (Add-on)
  - NLP to query data
  - Personal training required
  - Manually set up topics associated with datasets
    - How to handle dates
    - Datasets must be NLP-friendly
- Quicksight Paginated Reports
  - Can be based on Quicksight dashboards

## 266. Types of Visualizations, and When to Use Them

- Breakdown of Visualizations
  - Auto-chart
  - Chart types: bar, line, scatter, KPI, geospatial, word cloud

## 267. Amazon CloudTrail

- Breakdown of CloudTrail
  - By default enabled to provide audit of your AWS Account
  - Can send logs from CloudTrail into CloudWatch Logs or S3
    - CloudTrail expires in 90 days
  - All AWS API events are logged
    - Can separate Read events from Write events
    - Data events are not logged, such as S3 GetObject
    - CloudTrail Insights events
- CloudTrail Insights
  - Manually enabled to detect unusual activity
    - Hit Service limits
    - Inaccurate resource provisioning
  - Base line created by CloudTrail Insights

## 268. Amazon CloudTrail - Hands On

## 269. AWS Config

- Breakdown of Config
  - Auto-discovery of AWS resources (no free tier)
    - Evaluate resources against config rules (do not prevent)
  - Record configurations and changes over time
    - Is there unrestricted SSH access to my Security Groups?
    - Do my buckets have public access?
    - Has my ALB config change over time
  - AWS Config is a per region service
  - Config rules are basically for complaints or warning
    - Does not prevent the actions
- Config rules
  - AWS Managed config rules (over 140)
  - Create custom config rules (defined in AWS Lambda)
  - Rules can be triggered
    - For each config change
    - Scheduled intervals
- Config rules Automation
  - Auto-remediate using SSM Documents (max 5 retries)
  - Auto-notify using EventBridge

## 270. AWS Config - Hands On

## 271. CloudWatch vs CloudTrail vs Config

## 272. AWS Budgets

- Breakdown of Budgets
  - Create budget and send alarms when actual cost exceeds budget
    - Set up to 5 SNS notifications per budget
    - Filter by AWS resources
    - Attach an action to alarm (requires IAM permissions)
  - Budget types: Usage, Cost, Reservation, Savings Plan
  - For Reserved Instance (RI)
    - Track utilization
    - Supports EC2, RDS, Redshift, etc
  - First two budgets are free

## 273. AWS Budgets - Hands On

## 274. AWS Cost Explorer

- Breakdown of Cost Explorer
  - Visualize a breakdown of AWS cost by group
  - Create custom reports at a high level or granular level
  - Auto-recommend an optimal Savings Plan (alternative to RI)
  - Auto-forecast usage up to 12 months based on historical usage

## 275. AWS Trusted Advisor

- Breakdown of Trusted Advisor
  - Auto-assess AWS resources in your AWS account on six categories
    - Cost optimization (not free)
    - Performance (not free)
    - Security (basic)
    - Fault tolerance (not free)
    - Service limit (basic)
    - Operational excellence (not free)
  - Fully managed config rules that checks for compliance (do not prevent)
  - Business Support plan for full set of assessment
    - Basic plan free for Security and Service limit only
