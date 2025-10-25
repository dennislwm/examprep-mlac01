# Generative AI Model Fundamentals
## 166. Intro: Generative AI Model Foundations

## 167. The Transformer Architecture

- Breakdown of Transformer Architecture
  - Last hidden state gets propagated into the next loop via a feedback mechanism
    - Useful for modeling sequential time series and language processing
    - One giant vector (last hidden state) creates a bottle neck that start information may be lost
  - Self-attention solves this last hidden state issue
    - Breaks down one last hidden state to multiple hidden states
    - Attention weights (one per word / token)
    - Works better with differences in word sequence
    - Starts to have a concept of relationships between words / tokens
    - RNNs are still sequential and cannot parallelize it
  - Transformers solves the parallelization issue
    - Replace RNNs with feed-forward neural networks (FFNNs)

## 168. Self-Attention and Attention-Based Neural Networks

- Breakdown of Self-Attention Neural Networks
  - Each encoder / decoder has a list of embeddings (vectors) for each token
  - Self-attention produces a weighted average of all token embeddings
    - Tokens with attention weights are tied to each other that are contextually similar
    - A new embedding that captures its meaning in context
  - Three matrices of weights (Wq, Wk, Wv) that are learned through back-propagation
    - Wq - Query
    - Wk - Key
    - Wv - Value
  - Every token gets a weight by multiplying its embedding against these 3 matrices
    - Every token has three different matrices of weights (Wq, Wk, Wv).
    - Compute a score for each token by dot product
        □ Each Token Wq multiplied with all tokens Wk (including its own Wk)
        □ Each Token Wv multiplied with its score to create a new vector Wz
        □ Softmax is applied to the scores to normalize them
        □ Sum all vectors Wz for each Token
        □ Repeat for each token in parallel
  - Masked Self-Attention
    - A mask can be applied to prevent tokens from peeking into future tokens
    - Each Token Wq multipled with all tokens Wk (except future tokens)
    - GPT uses masked self-attention to mimic human processing
  - Multi-Headed Self-Attention
    - Optimize Wq, Wk, Wv vectors into matrices
    - The number of rows (heads) in the matrices are processed in parallel

## 169. Applications of Transformers

- Breakdown of Applications of Transformers
  - ChatGPT - underlying transformer model (T stands for Transformer)
  - Translator app
  - Text classification, e.g. sentiment analysis
  - Named entity recognition
  - Text summarization
  - Code generation
  - Chatbot / Customer service app

## 170. Generative Pre-Trained Transformers: How They Work, Part 1

- From Transformers to Generative Pre-Trained Transformer (GPT-2)
  - GPT-2 consists of a stack of decoder blocks only
    - In contrast, BERT consists of a stack of encoder blocks only
    - Each decoder block consists
        □ A masked self-attention layer
        □ A Fast-Forward Neural Network
  - No concept of input or output as there is no correct result
    - All it does is generate the next token over and over
    - Using self-attention to maintain relationships to previous tokens
    - Training on unlabeled block of text rather than optimizing for a specific task
  - Hundreds of billions of parameters
    - (GPT-4 has over one trillion parameters)
  - GPT layers
    - Input processing of text
        □ Token Encoding - convert words into tokens
          ® LLMs may be used to split words into tokens
        □ Token Embedding - convert tokens into vectors
        □ Positional Encoding - position of a token relative to other tokens
          ® Uses an interleaved sine function so it works on any length
    - Stack of decoder blocks
        □ Each decoder block results in a more refined token embedding
    - Output processing of text
        □ Token Embedding - dot product of output vector and matrix
          ® Outputs probabilities (logits) for each token
        □ Token Probability - picks highest probability
          ® Set temperature to randomize picks

## 171. Generative Pre-Trained Transformers: How They Work, Part 2

- Appended to previous section (171).

## 172. LLM Key Terms and Controls (Tokens, Embeddings, Temperature, etc)

- Breakdown of LLMs Key Terms
  - Token - numerical representation of words or parts of words
  - Embeddings - encode the meaning of a token into a vector
  - Context window - number of tokens an LLM can process at once
    - How many tokens that surround each token for self-attention
  - Max tokens - Limit on the number of input or output tokens
- Output controls for Token Probability layer
  - Temperature - randomize picks based on user setting (0 to 1)
    - Higher means more random
  - Top P - threshold probability for token to be included in output
    - Higher means more random
  - Top K - token to be included for K candidates in output
    - Higher means more random (more choices as K increases)

## 173. Fine-Tuning and Transfer Learning with Transformers

- Breakdown of Transfer Learning (Fine-Tuning) with GPT
  - Use any GPT model as a starting model for transfer learning (fine-tuning)
  - Add additional custom training data
  - Methods of transfer learning
    - Freeze specific layers, re-train others
        □ For example, train a new tokenizer to learn a new language
    - Add a layer on top of the GPT model
        □ For example, provide prompts and completions pairs

## 174. Lab: Tokenization and Positional Encoding with SageMaker Notebooks

## 175. Lab: Multi-Headed, Masked Self-Attention in SageMaker

## 176. Lab: Using GPT within a SageMaker Notebook

## 177. AWS Foundation Models and SageMaker JumpStart with Generative AI

- Breakdown of AWS Foundation Models
  - Generative Pre-Trained Transformer  (GPT) models
  - Llama Text (Meta)
  - Segment Anything Multi-Modal (Meta)
  - Titan Text (Amazon)
  - Nova (Amazon)
  - Alexa Multilingual Text (Amazon)
  - Jurassic-2 Multilingual Text (AI21labs)
  - Claude Text (Anthropic)
  - Stable Diffusion Multi-Modal (Stability.ai)
- SageMaker JumpStart
  - Deploy an AWS Foundation (GPT) model as a notebook
  - Integrated with HuggingFace

## 178. Lab: Using Amazon SageMaker JumpStart with Huggingface

