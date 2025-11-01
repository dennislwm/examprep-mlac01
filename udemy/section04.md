# AWS Managed AI Services
## 88. Intro: AWS Managed AI Services

- Short introduction of AWS Managed ML models

## 89. Why AWS AI Managed Services?

- Breakdown of AWS AI Managed Services
  - Pre-trained ML models for your use case
  - Documents: Comprehend, Translate and Textract
  - Images: Rekognition
  - Search: Kendra
  - Chatbots: Lex
  - Speech: Poly and Transcribe
  - Recommendations: Personalize
- Why use AI Managed Services
  - HA and responsiveness
  - Performance using GPUs
  - Token-based pricing for PAYG
  - Provisioned throughput for predictable workload

## 90. Amazon Comprehend

- Amazon Comprehend
  - ML model for NLP
  - Find insights and relationships in text
  - Extract key phrases, people, etc with NER
  - Detect PII
  - Understand text sentiment
  - Auto-organization of text files by topic
  - Use cases: analyze emails, group files by topic
  - Custom classification, e.g. support, complaints, etc
  - Supports multiple file formats, e.g. PDF, Word, etc
  - Named Entity Recognition (NER)
    - Person, Organisation, Location, Date, etc with confidence levels
  - Custom Entity Recognition (CER)
    - Train to recognise, such as Invoice, Policy numbers, etc
  - Share custom models with other AWS accounts
    - Configure a resource based IAM policy in your account
    - Use the ImportModel API operation in the other AWS accounts to securely import the custom model

## 91. Amazon Comprehend Custom Models

- Breakdown of Amazon Comprehend Custom Models
  - Train to recognise CER on your own data
  - Fully managed model versioning
  - Copied custom models across AWS accounts within the same region

## 92. Amazon Comprehend - Hands On

- Breakdown of Amazon Comprehend - Hands On
  - Insights
    - NERs - Name, date, etc
    - Key phrases
    - Language - English, etc
    - PII - Name, mobile, etc
    - Sentiment - Positive, negative
    - Targeted sentiment
    - Syntax - verbs, nouns, etc
  - Custom classification
    - Input data for training as CSV file
    - Purchase endpoint for real-time analysis using your custom model
    - Batch analysis for asynchronous jobs using your custom model
    - Use cases: classify a file by user topic

## 93. Amazon Translate

- Breakdown of Amazon Translate
  - Supports many languages
  - Easily translate large volumes of text efficiently

## 94. Amazon Translate - Hands On

- Breakdown of Amazon Translate - Hands On
  - Text
  - Document. Such as txt, docx, or html
  - Batch translation
    - S3 bucket input
    - S3 bucket output
    - Metrics
  - Custom terminology
    - Create a custom dictionary in CSV format
  - Parallel data
    - Influence translation for specific terms, such as law or finance

## 95. Amazon Transcribe

- Breakdown of Amazon Transcribe
  - Convert speech to text using automatic speech recognition (ASR)
  - Auto-remove PII using redaction
  - Use cases: customer service calls, closed captioning, meeting notes, etc
  - Improve accuracy by training own model or by adding a custom vocabularies
  - Custom Vocabularies for words
    - Add specific phrases, words, brand names
    - Providing hints
  - Custom Language Models for context
    - Provide your own specific text data
    - Learn the context for a given phrase or word
  - Combined both Custom Vocubularies and Custom Language Models for highest accuracy
  - Auto-detection of toxicity by tone and pitch, and text-based cues
    - Many categories, such as harassment, hate, etc

## 96. Amazon Transcribe - Hands On

- Breakdown of Amazon Transcribe - Hands On
  - Real-time transcribe
    - Auto-remove PII
    - Auto-identify language, e.g. French, English

## 97. Amazon Polly

- Breakdown of Amazon Polly
  - Convert text to speech using deep learning
  - Lexicons - specific text to speech
    - For example, AWS is Amazon Web Services
  - SSML (Speech Synthesis Markup Language)
    - For example, <break> for a short pause during speech
  - Many voice engines, such as generative, long-form, neutral, standard, etc
  - Speech mark for encoding where a word starts or ends in an audio
  - Voice types, e.g Ruth female, Gregory male, etc

## 98. Amazon Polly - Hands On

- Demo of Amazon Polly - Hands On

## 99. Amazon Rekognition

- Breakdown of Amazon Rekognition
  - Insights into images or videos using ML
    - Faces
    - Objects
    - Text extraction
    - Personal Protective Equipment (PPE) detection, e.g. face mask, helmet, etc
    - Scenes
    - Celebrities
    - Physics
    - Content moderation
        □ Integrates with Amazon Augmented AI for human review
        □ Custom moderation adaptors, similar to custom labels
  - Custom labels, e.g brand names
    - Input data for training as CSV file
    - Purchase endpoint for real-time analysis using your custom model
    - Batch analysis for asynchronous jobs using your custom model
    - Use cases: classify an image with brand names

## 100. Amazon Rekognition - Hands On

- Demo of Amazon Rekognition - Hands On

## 101. Amazon Lex

- Breakdown of Amazon Lex
  - Integrates with AWS Lambda, Comprehend, Kendra, Connect
  - Bot will understand user intent and invoke the correct Lambda function, e.g. MCP server
  - Custom voice or text chatbots for your applications
    - Use cases: booking status, etc

## 102. Amazon Lex - Hands On

- Demo of Amazon Lex - Hands On
  - Creation method by Traditional or Generative AI
  - Security: IAM
  - Voice or text interaction
  - Visual builder for a custom intent
    - Default intent if no intent detected
    - Input context
    - Output context
    - Sample utterances
    - Bot prompts for slots
    - Confirmation
    - Fulfillment using a Lambda function

## 103. Amazon Personalize

- Breakdown of Amazon Personalize
  - Same technology used by Amazon.com
  - Recipes are algorithms for specific use cases
    - Must provide training configuration for each recipe
  - Custom recommendation ML model
    - Input S3 data for training as CSV file
    - Purchase endpoint for real-time analysis using your custom model
    - Batch analysis for asynchronous jobs using your custom model
    - Use cases: song recommendation, movie, etc

## 104. Amazon Textract

- Breakdown of Amazon Textract
  - Auto-extract text, handwriting, and data from any scanned document
  - Use cases: invoices, medical records, passports, etc

## 105. Amazon Textract - Hands On

- Breakdown of Amazon Textract - Hands On
  - Analyze document
  - Analyze expense
  - Analyze ID
  - Analyze Lending
  - Bulk Document Uploader
  - Custom queries using natural language

## 106. Amazon Kendra

- Breakdown of Amazon Kendra
  - Auto-search answers from a document using natural language
  - Auto-indexing by Kendra
  - Incremental learning from user interactions

## 107. Amazon Augmented AI (A2I)

- Breakdown of Amazon Augmented AI
  - A gateway that sits between your ML model and the client application
    - Custom ML model can be hosted on AWS or another provider
    - Textract key-value pairs extracted by Amazon Textract
    - Rekognition image moderation
  - High-confidence predictions are returned immediately to the client application
  - Low-confidence predictions are sent for human review
    - Reviews are consolidated using weighted scores stored in Amazon S3
    - Reviewed data can be added to the training dataset to improve the ML model
  - Similar to Amazon Ground Truth in terms of human inputs
    - Reviews for predictions in Amazon Augmented AI
    - Labels for data in Amazon Ground Truth

## 108. Amazon Augmented AI - Hands On

- Breakdown of Amazon Augmented AI - Hands On
  - Conditions for invoking human review
    - Randomly send a sample
    - Threshold with a confidence score
  - Worker task template
    - Instructions for human review
  - Worker types
    - Amazon Mechanical Turk
    - Private team
    - Vendor third-party from Amazon Marketplace

## 109. Amazon's Hardware for AI

- Breakdown of Amazon's Hardware for AI
  - EC2 compute, sizing
    - GPU-based P3-P5, G3-G6
    - AWS Trainium TRN1, cost reduction when training
    - AWS inferentia INF1-INF2, cost reduction when inference
  - EBS or EFS storage, sizing
  - ELB network
  - ASG scaling
  - Security group

## 110. Amazon's Hardware for AI - Hands On

- Demo of Amazon's Hardware for AI - Hands On

## 111. Amazon Lookout

- Breakdown of Amazon Lookout
  - Auto-detect anomalies or unusual variances
  - Setup a detector to monitor datasets to find anomalies
  - The features that influence it are dimensions, and you optimize the measure
- Amazon Lookout for Equipment
  - Integrate with equipment sensors for predictive maintenance
- Amazon Lookout for Metrics
  - Input from AWS S3, Redshift, etc
  - Notification to SNS, Lambda
  - Feedback to Metrics for fine-tuning
- Amazon Lookout for Vision
  - Automate quality inspection with computer vision

## 112. Amazon Fraud Detector

- Breakdown of Amazon Fraud Detector
  - Auto-detect frauds customized to your data
  - Auto-insights on the importance of your feature
  - Integration with SageMaker
- Custom fraud ML model
  - Input S3 data for training as CSV file
  - Purchase endpoint for real-time analysis using your custom model
  - Batch analysis for asynchronous jobs using your custom model
  - Use cases: online payments, trial account abuse, etc

## 113. Amazon Q Business

- Breakdown of Amazon Q Business
  - Generative AI assistant for your employees trained on your company's data
  - Built on Amazon Bedrock but cannot choose underlying Foundation Model
  - Fully managed RAG with 40+ Data Connectors, such as S3, Aurora, Gmail, Slack, etc
  - Plugins or webhooks to interact with third-party services, such as Jira, ServiceNow, etc
  - Security: IAM Identity Center, Admin controls guardrails
- Custom generative AI model
  - Input S3 data for training as CSV file
  - Purchase endpoint for real-time analysis using your custom model
  - Use cases: answer questions, provide summaries, generate content, automate tasks

## 114. Amazon Q Business - Hands On

- Demo of Amazon Q Business - Hands On

## 115. Amazon Q Apps

- Breakdown of Amazon Q Apps
  - Generative AI powered apps without coding using prompt engineering

## 116. Amazon Q Apps - Hands On

- Demo of Amazon Q Apps - Hands On

## 117. Amazon Q Business - Hands On - Cleanup

- Demo of Amazon Q Business - Hands On - Cleanup

## 118. Amazon Q Developer

- Breakdown of Amazon Q Developer
  - Generative AI for AWS resources in your account
  - Suggest CLI command to run to make changes in your account
  - Analyze bills, and perform troubleshooting
  - AI code assistant similar to Claude Code
    - Supports many languages such as Python, Java, etc
  - Integrates with IDE such as VS Code, Jetbrains, etc

## 119. Amazon Q Developer - Hands On

- Demo of Amazon Q Developer - Hands On
