# Building Generative AI Applications with Bedrock
## 179. Intro: Building Generative AI Applications with Bedrock

## 180. Building Generative AI with Bedrock and Foundation Models

- Breakdown of Bedrock and Foundation Models
  - Underlying LLM that applications can be built on
  - API for Amazon and third-party LLMs
    - Third-party billing through AWS
  - Bedrock API: manage, deploy, train models
  - Bedrock-runtime API: perform inferences, generate embeddings
    - Converse, InvokeModel
  - Bedrock-agent API: Manage, deploy, train LLM agents and knowledge bases
  - Bedrock-agent-runtime API: perform inferences against agents and KBs
    - InvokeAgent
  - Security: IAM user/role
  - Model Access
    - Request access for models before usage
    - Amazon Titan models will be approved immediately

## 181. Lab: Chat, Text, and Image Foundation Models in the Bedrock Playground

## 182. Fine-Tuning Custom Models and Continuous Pre-Training with Bedrock

- Breakdown of Fine-Tuning
  - Reduce on prompt engineering to save on tokens in the long run
  - Expensive to fine-tune model using your own data in the short run
  - Only some models can be fine-tuned
    - Amazon Titan
    - Cohere
    - Meta
  - Input JSON array of objects with prompt and completion.
    - [{ "prompt": "<user_prompt>", "completion": "<expected_response>" }]
  - Continued Pre-Training
    - Unlabeled training data
    - [{ "input": "<user_prompt" }]

## 183. Retrieval-Augmented Generation (RAG) Fundamentals with Bedrock

- Breakdown of RAG
  - You query some external database for answers instead of relying on the LLM
  - User prompts are augmented with external data before sending to LLM
  - RAG > Fine-Tuning
    - Update database > Update model
    - Leverage semantic search via vector stores
  - Sensitive to augmented prompts, higher cost due to extra tokens
    - How many similar vectors to augment user prompt?
    - How to phrase the vectors within prompt?
  - User prompt -> Compute vector (using Embedding LLM) -> Database - > Retrieve Similar Data -> Send augmented prompt to LLM -> response

## 184. Vector Stores and Embeddings with Bedrock Knowledge Bases

- Breakdown of Vector Stores and Embeddings
  - Data Store (instead of Database)
  - Knowledge Base (instead of RAG)
  - Choose appropriate Data Store for the type of data
    - Graph Data Store (Neo4j) for recommendations or relationships
    - TF/IDF Data Store (Amazon Opensearch) for general search
    - Hybrid Data Store
- Embedding (Vectors)
  - A multi-dimensional vector associated with your data
  - Chunking controls how many tokens are represented by each vector
    - Slicing by characters, words, graph, etc
  - Similar data are close to each other in the multi-dim space
  - Embedding Models (Titan) can compute mass data
  - Your data is stored alongside the computed vectors in a vector database

## 185. Implementing RAG with Bedrock

- Breakdown of Implementing RAG
  - Your data: S3, Confluence, etc
  - Embedding LLM: Amazon Titan, Cohere, etc
  - Data Store: OpenSearch, MemoryDB, Aurora, MongoDB Atlas, etc
- Using Knowledge Base (RAG)
  - Chat with your document: Auto-RAG with your document
  - API to your RAG: invoked by your applications
  - Agentic RAG

## 186. Lab: Building and Querying a RAG System with Bedrock Knowledge Bases

## 187. Addendum: New chunking strategies in Bedrock

- In addition to fixed sized chunking, AWS has added additional choices
  - Semantic chunking - uses Foundation Model to group relevant sentences
  - Hierarchical chunking - uses parent-child chunks
  - Lambda-based - BYO chunking via Lambda

## 188. Content Filtering with Bedrock Guardrails

- Breakdown of Guardrails
  - Works with text Foundation models
  - Auto-filter by word or topic
    - Managed list of profanities
    - Add custom list of profanities
    - PII removal or masking
        □ Managed list of PII types
        □ Add custom regex of PII types
  - Contextual grounding check (sanity check)
    - Reduce hallucination
    - Measure how similar the response is to the contextual data received
  - Integrated with Agents and KBs

## 189. Lab: Building and Testing Guardrails with Bedrock

## 190. Building LLM Agents / Agentic AI with Bedrock Agents

- Breakdown of Agents
  - Agent decides which tools to use for what purpose
    - Lambda function in Bedrock
    - Action Group defines a tool
    - A prompt informs the Foundation Model when to use it
    - Defined the parameters that your Lambda function expects
  - Agent has a memory, typically a chat history or external data stores
  - Agent plans using a guidance on how to break down a request into sub-tasks
  - Enable code Interpreter for an Agent to write its own code
- Deploy Agent
  - Create an endpoint (alias ID) for an Agent
  - Agent type: On-Demand Throughtput or Provisioned Throughput
  - You application uses the InvokeAgent with alias ID

## 191. Lab: Build a Bedrock Agent with Action Groups, Knowledge Bases, and Guardrails

## 192. Other Bedrock Features (Model Evaluation, Studio, Watermarks)

- Breakdown of Other Bedrock Features
  - Import custom Foundation Model from S3 or SageMaker
  - Auto-evaluation of model: accuracy, BERTScore, F1, etc
    - Test against your own custom prompts or buil -in prompts
    - AWS managed team or BYO team to evaluate model
  - Detects if an image was generated by Titan
- Bedrock Studio
  - Web app IDE for Bedrock without requiring AWS accounts
  - Uses SSO integration with IAM and your own IDP
  - Collaborates on projects and components
