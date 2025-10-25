# Security, Identity, and Compliance
## 229. Intro: Security, Identity, and Compliance

## 230. Principle of Least Privilege

- Breakdown of Principle of Least Privilege
  - Grant only permissions required to perform a task
  - Start with a broader permission while developing
  - Lock down for production
  - Can use IAM Access Analyzer to generate a policy based on access activity

## 231. Data Masking and Anonymization

- Breakdown of Data Masking and Anonymization
  - Dealing with PII or sensitive data
  - Masking obfuscates data
    - Use cases: credit card, passwords
    - Supported by AWS Glue DataBrew, AWS Redshift
  - Anonymization replaces data
    - Replace with random text
    - Shuffle values in a column
    - Encrypt before storing data
    - Hashing data
  - Delete or don't import it

## 232. SageMaker Security: Encryption at Rest and In-Transit

- Breakdown
  - Set up user accounts with IAM with MFA
  - Set up roles to SageMaker services
  - Use SSL/TLS for all traffic
  - Use CloudTrail to log user activity
  - Use encryption before storing sensitive data
  - Use KMS for encrypting data at rest
    - Supported by SageMaker jobs, notebooks, S3
  - Optionally encrypt inter-node training layers
    - Increased training time and cost with deep learning
    - Enabled via console or API

## 233. SageMaker Security: VPC, IAM, Logging and Monitoring

- Breakdown
  - Supports training jobs in a private VPC
    - Need to setup S3 VPC endpoints
    - Use endpoint policy and S3 policy to keep data secure
  - Notebooks are public by default
    - Insecure but Internet can be disabled
    - If disabled, you will need a NAT Gateway or PrivateLink interface endpoint for training jobs
  - Training and inference containers are public by default
    - Need to setup S3 VPC endpoints
- SageMaker policy
  - IAM permissions
    - CreateTrainingJob
    - CreateModel
    - CreateEndpointConfig
    - CreateTransformJob
    - CreateHyperParameterTuningJob
    - CreateNotebookInstance
    - UpdateNotebookInstance
  - Predefined policy
    - AmazonSageMakerReadOnly
    - AmazonSageMakerFullAccess
    - AdministratorAccess
    - DataScientist
- SageMaker logging
  - CloudWatch can log, monitor and alarm on
    - Invocations and latency on endpoints
    - EC2 instances
    - GroundTruth active workers, idle
  - CloudTrail record users, roles and services activities
    - Logs are delivered to S3 for audit

## 234. IAM Introduction: Users, Groups, Policies

- Breakdown
  - Global IAM API - no region specified
  - Root account by default, shouldn't be used or shared
  - Groups can only contain users, but not sub-groups
    - Users can belong to multiple groups or no groups
  - Group policy > User policy

## 235. IAM Users & Groups - Hands On

- Hands On to IAM Users & Groups

## 236. AWS Console Simultaneous Sign-In

- Breakdown
  - AWS console supports multi-session
    - Multiple IAM user sessions in the same browser

## 237. IAM Policies

- Breakdown
  - IAM Policy Inheritance
    - IAM users inherit policy from multiple groups
  - IAM Policy JSON structure
    - Version
    - ID - optional identifier
    - Statement - array
        □ SID - optional statement identifier
        □ Effect - ["Allow" | "Deny"]
        □ Principal - user, role applied to
        □ Action - array
          ® object:action
        □ Resource - resources applied to

## 238. Iam Policies - Hands On

- Hands On to IAM Policies

## 239. IAM MFA

- Breakdown
  - User password policy
    - IAM default or custom policy
    - Minimum password length (min 6)
    - Require specific characters
    - Password expiration
    - Prevent password reuse
  - Enable Multi-Factor Authentication (MFA)
    - If password is stolen, the account is not compromised
    - Can register up to 8 MFA devices per account
    - Virtual MFA device
        □ Google Authenticator
        □ Authy
        □ Lastpass Authenticator
    - Universal 2nd Factor (U2F) Security Key
        □ YubiKey
        □ Supports multiple users per key
    - Hardware Key Fob
        □ Gemalto
        □ SurePassID for AWS GovCloud

## 240. IAM MFA - Hands On

- Hands On to IAM MFA

## 241. IAM Roles

- Breakdown
  - Users - person to service; Roles - service to service

## 242. IAM Roles - Hands On

- Hands On to IAM Roles

## 243. Encryption 101

- Breakdown
  - Encryption in flight (TLS/SSL)
    - Data is encrypted before sending; Data is decrypted after receiving
    - TLS certificates used with HTTPS
  - Encryption at rest
    - Server-side encryption (SSE)
        □ Data is encrypted after receiving by the server; Data is decrypted before being accessed
        □ Server must have access to key stored somewhere
    - Client-side encryption
        □ Data is encrypted and encrypted by client before being sent to / read from the server
        □ Client must have access to key

## 244. AWS KMS

- Breakdown
  - Able to audit KMS key using CloudTrail
  - Seamless integration with most AWS services
  - AWS managed encryption keys
  - KMS scoped per region
    - Allow another account to access Key A
    - Encrypt EBS snapshots with Key A in Region A
    - Copy EBS from Region A to Region B
    - Decrypt with Key A in Region B
    - Encrypt EBS with Key B in Region B
- Key Policy
  - Default policy
  - Custom policy
    - Required for migration using customer managed key
- KMS types
  - Symmetric (AES-256)
    - Single encryption key to encrypt and decrypt
    - Must call KMS API to use
  - Asymmetric key (RSA)
    - Pair of encryption keys to encrypt (public) and decrypt (private)
    - Used by users who don't have access to KMS API
  - AWS Owned keys
    - SSE-S3, SSE-SQS, SSE-DDB (default)
  - AWS Managed keys
    - Free
    - AWS/S3
    - AWS/RDS
    - Auto-rotation
  - Customer Managed Keys created within KMS: $1/month
    - Manual rotation by default
    - Auto-rotation must be enabled
  - Customer Managed Keys imported to KMS: $1/month
    - Manual rotation

## 245. AWS KMS - Hands On

- Hands On to AWS KMS

## 246. Amazon Macie

- Breakdown
  - AWS fully-managed service to discover and protect PII and sensitive data
    - Uses ML and pattern matching
  - S3 data analyzed by Macie
  - Event notify using EventBridge by Macie

## 247. AWS Secrets Manager

- Breakdown
  - Secret expiration can be enabled
  - Auto-rotation can be enabled using Lambda
  - Encrypted using KMS key
  - Integrated with RDS
  - Multi-region secrets are synced across regions
    - Can promote a read replica secret to a standalone secret
    - Used for disaster recovery

## 248. AWS Secrets Manager - Hands On

- Hands on to AWS Secrets Manager

## 249. AWS WAF

- Breakdown
  - Protection at Layer 7 (Application layer) via HTTPS
  - Deploy targets
    - Application LB
        □ NLB not supported
    - API Gateway
    - CloudFront
    - AppSync GraphQL API
    - Cognito User Pool
- Web Access Control List (ACL)
  - Set IP addresses
  - Set HTTP headers
  - Size constraints
  - Geo-block
  - Rate-based rules for DDoS protection
- Web ACLs are per region
  - Exception for CloudFront (global)
  - ALB must be in the same region

## 250. AWS Shield

- Breakdown
  - Protection from DDoS protection
  - AWS Shield Standard by default
    - Active for every AWS account
    - Basic protection
- AWS Shield Advanced
  - $3,000 per month per organization
  - Advanced protection
  - 24/ access to Response team
  - Auto-create, evaluate and deploy AWS WAF rules

## 251. VPC, Subnets, Internet Gateway, NAT Gateway

- Breakdown
  - VPC is a logical network within a region
  - Subnet is a logical network within an AZ
  - Route tables define access to Internet and between subnets
- Internet Gateway
  - Attach to a subnet to connect to the Internet
  - Any subnet attached is a public subnet
- NAT Gateway (AWS-managed) / NAT Instances (Client-managed)
  - Requires to be in the same subnet as Internet Gateway
  - One-way access to the Internet
  - Any subnet attached to NAT is a private subnet

## 252. NACL, Security Groups, VPC Flow Logs

- Network ACLs
  - Virtual firewall attached to a subnet
  - Allow or deny rules
    - IP addresses
    - No other criteria
  - By default allow all traffic
  - Stateless
    - Return traffic not allowed unless specified
- Security groups
  - Virtual firewall attached to an EC2 instance
  - Allow rules ONLY
    - IP address
    - Other security groups
  - By default deny all traffic
  - Stateful
    - Return traffic allowed always
- VPC Flow Logs
  - Network audit of all traffic for IPs in CIDR range of VPC
  - Integrates with S3, CloudWatch, Kinesis

## 253. VPC Peering, Endpoints, VPN, Direct Connect

- VPC Peering
  - Connect TWO (2) logical network partitions
  - Make both network partitions behave as one partition
  - Must not have overlapping CIDR range
  - Not transitive
    - Assume A and C has a VPC peering
    - Creating a VPC peering between B and C logical network partitions does not auto-connect A and C partitions
- VPC Endpoints
  - Allows traffic to flow within AWS private network instead of using public Internet
  - AWS services are public by default
  - VPCE Gateway
    - S3, DynamoDB
  - VPCE Interface
    - S3, DynamoDB
    - Most AWS services
- Connect a logical network partition with on-premises network
  - Site-to-site VPN
    - Auto-encrypted traffic using VPN
    - Traffic to flow using public Internet
  - Direct Connect (DX)
    - Requires a physical connection
    - Traffic to flow using AWS private network

## 254. VPC Cheat Sheet & Closing Comments

## 255. AWS PrivateLink

- Breakdown
  - AWS PrivateLink
    - AWS Private Link exposes a service rather than making both network behave as one (VPC Peering)
    - AWS Private Link supports third-party services rather than AWS services (VPCE Gateway / Interface)
  - Expose a third-party vendor service as an endpoint within a logical network partition
    - Vendor creates a NLB attached to their service
    - Client creates an ENI within a logical network partition
    - Traffic flows from the vendor's NLB to the client's ENI within AWS private network
