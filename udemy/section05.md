# SageMaker Built-In Algorithms
## 120. Intro: SageMaker Built-In Algorithms

- Short intro of [SageMaker Built-In Algorithms][r04]

## 121. Introducing Amazon SageMaker

- Breakdown of Introducing Amazon SageMaker
  - Built to handle entire ML workflow
  - Managed compute when training and deploying a ML model
  - Managed auto-scaling endpoint when deploying to client app
- SageMaker Notebook
  - Notebook instances on EC2 that are spun up from the console
  - Ability to spin up training instances and deploy trained models

## 122. SageMaker Input Modes

- Breakdown of SageMaker [Input Modes][r01]
  - Default S3 File Mode
    - Slow: copy training data from S3 to local directory in Docker container
  - S3 Fast File Mode
    - Limitations
      - Data must reside in S3, and not with Amazon FSx for Lustre
      - Suited for workloads with many small files
    - Fast: training can begin without waiting to copy entire data
  - S3 Pipe Mode
    - Supports high concurrency and throughput
    - Suited for workloads with large files
    - Stream: streams data directly from S3
  - Others
    - S3 Express One Zone for high performance in a single AZ
    - FSx for Lustre to scale 100GB+ of throughput and miillions of IOPS with low latency in a single AZ, VPC
    - EFS requires VPC

## 123. Linear Learner in SageMaker

- [Linear Learner (LL) in SageMaker][r02]
  - Specifically designed to handle class imbalance by adjusting class weights
  - Supervised learning ideal for classifications or predictions
  - Requires less hyperparamater tuning as compared to other complex algorithms
  - Linear regression: for numeric predictions based on a fitted linear equation
  - Linear threshold: for classifications (binary or multi-class)
  - Input: CSV first column assumed to be the label
- Preprocessing
  - LL can auto-normalize training data so all features are weighed the same
  - Input data should be shuffled
- Training
  - LL uses stochastic gradient descent (SGD)
  - User can choose optimization algorithm, such as Adam, AdaGrad, SGD, etc
  - Multiple models are optimized in parallel
  - Hyperparameters: tune L1, L2 regularization
    - [Balance multiclass weights][r03]
      - Useful for imbalanced datasets where certain classes have fewer data
      - Assigns higher weights to under-represented classes
    - Learning rate
      - Large learning rate can lead to faster convergence but may cause the model to overshoot the optimal solution
      - Small learning rate can lead to slower convergence but may result in a more stable model
    - Mini batch size
      - Large batch size can lead to poor model generalization with faster training
      - Small batch size can lead to improve generalization but at the expense of slower training
    - L1 regularization
    - Weight decay (L2 regularization)
    - Target precision: Holds precision at this value while maximizing recall
    - Target recall: Holds recall at this value while maximizing precision
- Validation
  - Most optimal model is selected
- Use case: Hand writing recognition (Linear threshold function for classification of letters or digits)

## 124. XGBoost in SageMaker

- Breakdown of XGBoost in SageMaker
  - Extreme Gradient Boosting: boosted group of decision trees, where new trees are made to correct the errors of previous trees
  - Can be used for classification and predictions (using regression trees)
  - Fast and has been used in lots of Kaggle competitions
  - Based on the open-source XGBoost
  - Input: CSV, Parquet, recordIO-protobuf
- Training
  - Can use as a framework within notebooks, e.g. Sagemaker.xgboost.
  - Or as a built-in SageMaker algorithm
  - Hyperparameters: there are a lot of them
    - Subsample: prevents overfitting
    - ETA: Step size shrinkage, prevents overfitting
    - Gamma: Minimum loss reduction to create a partition (larger means more conservative)
    - Alpha: L1 regularization term (larger means more conservative)
    - Lambda: L2 regularization term (larger means more conservative)
    - Eval_metric: Optimize on Area Under Curve (AUC) to reduce false positives
    - Scale_pos_weight: Adjusts balance of positive and negative weights, helpful for unbalanced classes
    - Max_depth: Maximum depth of tree, too high and you may overfit
  - Instance types: XGB is memory-bound, not compute-bound
    - M5 is a good choice
    - As of XGB 1.2, single-instance GPU using P2, P3 can be quicker for training and more cost effective
        □ Must set hyperparameter tree_method = gpu_hist
    - As of XGB 1.5+ distributed GPU can be quicker but only works with CSV or Parquet
        □ Must set use_dask_gpu = true; distribution = fully_replicated

## 125. LightGBM in SageMaker

- Breakdown of LightBGM in SageMaker
  - Similar to XGB, but even more like CatBoost
  - Extended with Gradient-based One-Side Sampling and Exclusive Feature Bundling
  - Predicts target variables with an ensemble of estimates from simpler models
  - Can be used for classification, regression or ranking
  - Input: CSV
- Training
  - Hyperparameters
    - Learning_rate
    - Num_leaves: Maximum leaves per tree
    - Feature_fraction: Maximum features represented per tree
    - Bagging_fraction: Similar to feature_fraction, but randomly sampled
    - Bagging_freq: How often bagging is done
    - Max_depth: Maximum depth of tree
    - Min_data_in_leaf: Minimum amount of data per leaf, can be used to address overfitting
- Instance types: Memory-bound algorithm
  - Choose general purposed over compute-optimized instances, e.g. M5 > C5
  - Single or multi-instance CPU training, no GPU instances
  - Define instances with instance_count

## 126. Seq2Seq in SageMaker

- Breakdown of Seq2Seq in SageMaker
  - Input a sequence of tokens, output a sequence of tokens
  - Implemented with RNN and CNN with attention
  - Input: RecordIO-Protobuf (tokens must be integers in vocabulary file)
- Use cases: machine translation, text summarization, speech to text
- Training
  - Hyperparameters
    - Batch_size
    - Optimizer_size: adam, sgd, rmsprop
    - Learning_rate
    - Num_layers_encoder
    - Num_layers_decoder
  - Quirks in optimization:
    - Accuracy: compares against provided validation dataset
    - BLEU score: compares against multiple reference translations
    - Perplexity: cross-entropy
- Instance types
  - Can only use a single GPU instance for training, eg P3
    - Cannot run in parallel, however a single instance can use multi-GPU

## 127. DeepAR in SageMaker

- Breakdown of DeepAR in SageMaker
  - Implemented with RNN
  - Allows you to train the same model over several related time series
- Input types
  - JSON format in Gzip or Parquet
  - Each record must contain
    - Start time stamp
    - Target time series value
  - Additional data include categories, dynamic features, etc
- Use cases
  - Forecasting one-dimensional time series data
  - Finds frequency and seasonality
- Hyperparameters
  - Context_length: number of time points before a prediction
  - Epochs
  - Mini_batch_size
  - Learning_rate
  - Num_cells
  - Quirks in training
    - Always include entire time series for training, testing, and inference
    - Train on many related time series
    - Use entire dataset as test set, remove last time points for training and evaluate on withheld values
- Instance types
  - Both CPU and GPU
    - CPU only for inferences
  - Single or multi instances
  - Recommend start with CPU, e.g. ml.c4.2xlarge
  - GPU only necessary for larger models and large Mini_batch_size

## 128. BlazingText in SageMaker

- Input types for text classification
  - One sentence per line
  - Prefix __label__<classifier_no> in each sentence
  - Spaces around each word, including punctuation
  - Can be text or json input
    - Json: "source" and "label" keys in each object
- Use cases
  - Text classification: predict labels for a sentences
  - Web searches or information retrieval
- Word2vec
  - Creates a vector representation of words
  - Semantically similar words are vectors close to each other
  - Also known as word embedding
  - Not useful for sentences
  - Mode types
    - Continuous Bag of Words (CBOW)
        □ No relationships between words, i.e. order of words is not important
    - Skip-gram
        □ Order of words is important
    - Batch skip-gram
        □ Distributed compute over many CPU nodes
- Hyperparameters
  - Word2Vec
    - Mode type: batch_skipgram, skipgram, cbow
    - Learning_rate
    - Window_size
    - Vector_dim
    - Negative_samples
  - Text classification
    - Epochs
    - Learning_rate
    - Word_ngrams
    - Vector_dim
- Instance types
  - For cbow and skipgram: any single CPU or GPU, i.e. ml.p3.2xlarge
  - For batch_skipgram: multiple CPU instances
  - For text classification: any single CPU or GPU (> 2GB training data)

## 129. Object2Vec in SageMaker

- Use cases
  - It's similar to Word2vec, except for arbitrary objects
  - It creates low-dim dense embeddings of high-dim objects
  - Compute nearest neighbors of objects, visual clusters
  - Recommendations for similar objects
- Input types
  - Objects must be tokenized into integers
  - Data consists of pairs of token
    - Product-product
    - Customer-customer
    - User-item
    - Genre-description
    - Sentence-sentence
  - Json: "label", "in0", and "in1" for each object
    - Label: expected class for the object
    - In0: Array of integers representing first object in the pair
    - In1: Array of integers representing second object in the pair
- Training
  - Process JSON lines and shuffle them
  - Pair of inputs, pair of encoders, and a single comparator
  - Comparator is followed by a Feed-Forward Neural Network, followed by a decoder
  - Encoder choices
    - Average-pooled embeddings
    - CNNs
    - Bidirectional LSTM
- Hyperparameters
  - Typical:
    - Dropout
    - early stop
    - Epochs
    - learning_rate
    - Batch_size
    - Layers
    - Activation_function
    - Optimizer
    - Weight_decay
  - Specific:
    - Enc1_network
    - Enc2_network
        □ HCNN
        □ BiLSTM
        □ Pooled_embedding
- Instance types
  - Train on a single instance, CPU or GPU or multi-GPU single instance
    - Ml.m5.2xlarge
    - Ml.p2.xlarge

## 130. Object Detection in SageMaker

- Use cases
  - Identify all objects in an image with bounding boxes
  - Detects and classifies objects with a single Deep NN
  - Classes are accompanied by confidence scores
  - Can train from scratch or use pre-trained models based on ImageNet
- Input types
  - MXNet:
    - RecordIO or image format (png or jpg)
    - Json: Each image should have an object
        □ "file": filename
        □ "image_size": width, height, depth
        □ "annotations": class_id, left, top, width, height
        □ "categories": class_id, name
- Training
  - Two types: MXNet and Tensorflow
  - MXNet
    - Uses a CNN with the Single Shot multibox Detector (SSD) algorithm
        □ The base CNN can be VGG-16 or ResNet-50
    - Transfer learning mode / incremental training
        □ Use a pre-trained model for the base network weights instead of random initial weights
    - Uses flip, rescale, and jitter to avoid overfitting
  - Tensorflow
    - Uses ResNet, EfficientNet, MobileNet models from the Tensorflow Model Garden
- Hyperparameters
  - Mini_batch_size
  - Learning_rate
  - Optimizer: Sgd, adam, rmsprop, adadelta
- Instance types
  - Uses a single instance GPU or multi-GPU for training
    - Multi-instances GPU
    - Ml.p2.xlarge, ml.p3.2xlarge, G4dn, G5
  - Uses CPU or GPU for inference
    - M5, P2, P3, G4dn

## 131. Image Classification in SageMaker

- Use cases
  - Assign one or more labels to an image, e.g. cat
  - Does not include bounding boxes
- Input types
- Training
  - Two types: MXNet and Tensorflow
  - MXNet
    - Full training mode
        □ Network initialized with random weights
    - Transfer learning mode
        □ Network initialized with pre-trained weights
        □ Network is fine-tuned with new training data
    - Default image is 3-channel 224x224 (ImageNet's dataset)
  - Tensorflow
    - Uses various Tensorflow Hub models: MobileNet, Inception, ResNet, EfficientNet
        □ Top classification layer is available for fine tuning
- Hyperparameters
  - Batch_size
  - Learning_rate
  - Optimizer
    - Weight decay, beta 1, beta 2, eps, gamma
    - Slight difference between MXNet and Tensorflow
- Instance types
  - Uses a single instance GPU or multi-GPU for training
    - Multi-instances GPU
    - Ml.p2.xlarge, ml.p3.2xlarge, G4dn, G5
  - Uses CPU or GPU for inference
    - M5, P2, P3, G4dn

## 132. Semantic Segmentation in SageMaker

- Use cases
  - Image pixel-level object classification that produces a segmentation mask
  - Different from image classification that assigns labels to whole images
  - Different from object detection that assigns labels to bounding boxes
  - Useful for self-driving vehicles, medical imaging diagnostics, robot sensing
- Input types
  - JPG and PNG images for both training and validation
  - JPG images for inference
  - Augmented manifest image format supported for Pipe mode
  - Label maps to describe annotations
- Training
  - Built on MXNet Gluon and Gluon CV
  - Choice of three algorithms
    - Fully-Convolutional Network (FCN)
    - Pyramid Scene Parsing (PSP)
    - DeepLabV3
  - Choice of backbones
    - ResNet50
    - ResNet101
    - Both trained on ImageNet
  - Both incremental and training from scratch supported
- Hyperparameters
  - Common
    - Epochs
    - Learning_rate
    - Batch_size
    - Optimizer
  - Specific
    - Algorithm
    - Backbone
- Instance types
  - Training: Only GPU instances are supported on a single machine
    - P2, P3, G4dn, G5
  - Inference: CPU (C5, M5) or GPU (P3, G4dn)

## 133. Random Cut Forest in SageMaker

- Use cases
  - Amazon algorithm for anomaly detection
  - Unsupervised training on time series data
    - Assigns a score to each data point
    - Works well for detecting fraud because it looks for unusual patterns in all of the data points
- Input types
  - RecordIO-protobuf or CSV
  - Both File or Pipe mode
  - Optional test channel for computing accuracy, precision, recall, F1 etc
- Training
  - Creates a forest of decision trees
    - Expected change in complexity of each tree as a result of adding a data point into it
        □ More changes to branches mean higher probability of anomaly data point
    - Data is sampled randomly, and each tree is a partition of the training data (Random Cut)
  - RCF integrates with Kinetic Analytics
- Hyperparameters
  - Specific
    - Num_trees: Increased reduces noise
    - Num_samples_per_tree
        □ Should be chosen such that 1/num_samples_per_tree approximates the ratio of anomalous data to normal data
- Instance types
  - Training: CPU only (M4, C4, C5)
  - Inference: CPU only (C5)

## 134. Neural Topic Model in SageMaker

- Use cases
  - Classify or summarize corpus of documents based on topics
  - Unsupervised means no labels required on data points
    - Group similar tokens
        □ For example, group tokens include bike, car, speed, train etc
        □ However, it does not label group and user has to determine what group label should be
    - Algorithm is neural variational inference
  - One of two top topic algorithm in SageMaker (the other LDA)
- Input types
  - Four data channels
    - Required train
    - Optional validation, test and auxiliary
  - RecordIO-protobuf or CSV
  - Words must be tokenized into integers
    - Every document must have a count for every word in the vocabulary in CSV
    - The auxiliary channel is for the vocabulary
  - Both File and Pipe mode supported
- Training
  - Define how many topics you want
- Hyperparameters
  - Common
    - Mini_batch_size
    - Learning_rate
  - Specific
    - Num_topics
- Instance types
  - Training: CPU (cheaper) or GPU (recommended)
  - Inference: CPU

## 135. Latent Dirichlet Allocation (LDA) in SageMaker

- Use cases
  - Can group words and other objects
    - Customers based on purchases
    - Harmonic analysis in music
  - Used for documents mostly
  - Unsupervised algorithm that works similar to Neural Topic
- Input types
  - Required train channel
  - Optional test channel
    - For scoring results, e.g. per-word likelihood
  - RecordIO-protobuf or CSV
  - Each document has counts for every word in vocabulary in CSV
  - File or Pipe mode (RecordIO only)
- Training
- Hyperparameters
  - Specific
    - Num_topics
    - Alpha0
        □ Initial guess for concentration parameter
        □ Smaller values generate sparse topic mixtures
        □ Larger values produce uniform mixtures
- Instance types
  - Training: CPU only on a single machine (cheaper than Neural Topic)
  - Inference: CPU only

## 136. K-Nearest-Neighbors (KNN) in SageMaker

- Use cases
  - Supervised (with labels) clustering
  - Simple classification
    - Find the K closest points to a sample point and return the most frequent label
  - Average regression
    - Find the K closest points to a sample point and return the average value
- Input types
  - RecordIO-protobuf or CSV
    - First column is label
    - File or pipe mode
- Training
  - Data is first sampled (subset of full training data)
  - SageMaker includes a dimensionality reduction
    - Avoid sparse data (curse of dimensionality)
    - At cost of noise / accuracy
    - Sign or fjlt methods
  - Builds an index for lookup neighbours
  - Serialize the model
  - Query the model for a given K
- Hyperparameters
  - K
  - Sample_size
- Instance types
  - Training: CPU or GPU instance
  - Inference: CPU (low latency) or GPU instance (high throughput)

## 137. K-Means Clustering in SageMaker

- Use cases
  - Unsupervised (without labels) clustering
  - Data is divided into K groups, where points in each group are "similar"
    - Similar based on Euclidean distance
- Input types
  - RecordIO-protobuf or CSV
    - No labels column
    - File or Pipe mode
  - Training channel:
    - ShardededByS3Key (data in S3 will be sharded to each training node)
  - Optional test channel:
    - FullyReplicated
- Training
  - Every observation is mapped to n-dim space (n=number of features)
  - Works to optimized the center of K clusters
    - Extra cluster (x) centers may be specified to improve accuracy (which end up getting reduced to k)
    - K = k*x
  - Algorithm
    - Determine initial cluster centers
        □ Random or k-means++ approach
        □ K-means++ tries to make initial clusters far apart
    - Iterate over training data and calculate cluster centers
    - Reduce clusters from K to k
        □ Using Lloyd's method with K-means++
- Hyperparameters
  - Choosing K is tricky
    - Optimize for tightness of clusters
    - Elbow method: plot within cluster sum of squares as function of K
  - Mini_batch_size
  - Extra_center_factor
  - Init_method
- Instance types
  - Training: CPU or GPU instance (only single GPU)
  - Inference: CPU or GPU instance

## 138. Principal Component Analysis (PCA) in SageMaker

- Use cases
  - Unsupervised dimensionality reduction technique
    - Renders higher-dim data into lower dimension data while minimizing loss of information
    - Reduced dimensions are called components
    - First component has largest possible variability, followed by second etc
- Input types
  - RecordIO-protobuf or CSV
  - Both File or Pipe mode
- Training
  - Covariance matrix is created, then singular value decomposition (SVD)
  - Two modes
    - Regular for sparse data and moderate number of observations and features
    - Randomized for large number of observations, uses approximation algorithm
- Hyperparameters
  - Specific:
    - Algorithm_mode
    - Subtract_mean useful for unbiasing data
- Instance types
  - Training: Both CPU or GPU supported

## 139. Factorization Machines in SageMaker

- Use cases
  - Supervised classification or regression algorithm for recommenders
  - Limited to pair-wise interactions on sparse data
    - User-item
    - Click prediction
    - Item recommendation
- Input types
  - RecordIO-protobuf with Float32
    - Sparse data means CSV is not practical
- Training
  - Finds factors we can use to predict a classification
    - Click or not?
    - Purchase or not?
  - Given a 2-dim matrix representing some pair-wise interaction
- Hyperparameters
  - Specific
    - Initialization methods for bias, factors and linear terms
        □ Uniform, normal or constant
        □ Can tune properties of each method
- Instance types
  - Training: CPU (recommended) or GPU (dense data only)

## 140. IP Insights in SageMaker

- Use cases
  - Unsupervised learning of IP address usage patterns
  - Identifies suspicious behavior from IP address
- Input types
  - CSV only
    - Entity (username/account_id), IP address
  - No preprocessing required
  - Training channel
  - Optional validation channel
    - Computes AUC score
- Training
  - Uses a NN to learn latent vector representations of Entities and Ips
  - Entities are hashed and embedded
    - Need large has size
  - Automatically generates negative samples during training by randomly pairing entities and Ips
    - Mitigate unbalanced data
- Hyperparameters
  - Num_entity_vectors
    - Hash size
    - Set to twice the number of unique entities
  - Vector_dim
    - Size of embedding vectors
    - Scales model size
    - Too large results in overfitting
  - Epochs
  - Learning_rate
  - Batch_size
- Instance types
  - Training: CPU or GPU instance (including multi-GPU instance)
    - Size of CPU depends on vector_dim and num_entity_vectors

## Links

[r01]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html
[r02]: https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html
[r03]: https://builder.aws.com/content/2eux4F6yvezbPPZFWnQbdHF9HwC/linear-learner-in-sagemaker-hyperparameter-tuning
[r04]: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html