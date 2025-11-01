# Model Training, Tuning and Evaluation
## 141. Intro: Model Training, Tuning and Evaluation

## 142. Introduction to Deep Learning

## 143. Activation Functions

- Breakdown of Activation Functions
  - Picking the right activation function can be important for the performance of your NN.
  - Simplest activation function is a linear function
      - Cannot perform backpropagation (for learning)
  - Binary step function
      - Cannot handle multiple outputs
      - Vertical slopes don't work well with calculus
  - Pick non-linear activation functions
      - Can create complex mappings between inputs and outputs
      - Allow backpropagation because they have a useful derivative
      - Allow for multiple layers
- Non-linear Activation Functions
  - Sigmoid / Logistic / TanH
      - Nice and smooth curve
      - Scales input from 0-1 (Sigmoid / Logistic) or -1 to 1 (TanH / hyperbolic tangent)
        - MaxOut > PReLU > Leaky ReLU > ReLU > TanH > Sigmoid
      - Changes slowly for high or low values
        - Vanishing gradient issue where numerical precision becomes a problem
      - Computationally expensive
  - Rectified Linear Unit (ReLU)
      - Easy to compute
      - Similar to binary except a linear slope instead of vertical slope
        - Positive numbers = linear slope
      - Zero or negative values
        - Dying ReLU issue where input <=0 becomes a problem
  - Leaky ReLU
      - Solved Dying ReLU issue
        - Uses a negative slope where input <=0
        - Arbitrary constant for negative slope
  - Parametric ReLU (PReLU)
      - Solved arbitrary constant for negative slope
        - Uses backpropagation to determine the constant for negative slope
        - Complicated and Your Mileage May Vary (YMMV)
  - Other ReLU variants
      - Exponential Linear Unit (ELU)
        - Negative slope is a non-linear slope
      - Swish from Google
        - Not from AWS
        - Mostly benefit a very deep network (>40 layers)
      - Maxout
        - Outputs the max value of the inputs
        - Doubles parameters that need to be trained, not often practical
- Final output-layer activation function Softmax
  - Used in a multi-class classification problem
  - Converts outputs to probabilities of each classification
  - Cannot produce more than one labels per node
      - Use Sigmoid for multiple labels per node

## 144. Convolutional Neural Networks

- Breakdown of CNNs
  - When you have data that doesn't neatly align into columns
  - They can find features that aren't in a specific spot
      - Like a stop sign in a picture
      - Or words within a sentence
      - Sentiment analysis
      - Machine translation
      - Sentence classification
  - They are "feature-location invariant" inspired by the biology of visual cortex
      - Local receptive fields are groups of neurons that only respond to a part of what your eyes see (subsampling)
      - They overlap each other to cover the entire visual field (convolutions)
      - They feed into higher layers that identify increasingly complex images
        - Some receptive fields identify horizontal lines, etc (filters)
        - These would feed into a layer that identify shapes
        - Which then feed into a layer that identify objects
        - For color images, extra layers for RGB
      - How do we know that's a stop sign?
        - Individual local receptive fields scan the image looking for edges, and pick up the edges of the stop sign in a layer
        - Those edges in turn get picked up by a higher level convolution that identifies the stop sign's shape and letters
        - This shape then gets matched against your pattern of known stop signs, also using strong red signal coming from your RGB layers
        - That information keeps getting processed upwards until the result reaches the final output layer
- Building a CNN with Keras / Tensorflow
  - Resource intensive (CPU, GPU and RAM)
      - Lots of hyperparameters
        - Kernel sizes, many layers, etc
  - Source data must be of appropriate dimensions
      - Width x length x color channels (one for each RGB)
      - Getting the training data is the hardest part (as well as storing and accessing it)
  - Conv2D layer type does the actual convolution on a 2D image
      - Conv1D and Conv3D also available - doesn't have to be image data
  - MaxPooling2D layers can be used to reduce a 2D layer down by taking the maximum value in a given block
      - For example, a 3x3 pixel will take the maximum value and replacing for that block
  - Flatten layers will convert the 2D layer to a 1D layer for passing into a flat hidden layer of neurons
  - Typical workflow: Conv2D -> MaxPooling2D -> Dropout -> Flatten -> Dense -> Dropout (to prevent overfitting) -> Softmax
  - Specialized CNN architectures
      - LeNet-5: Good for handwriting recognition
      - AlexNet: Image classification (deeper than LeNet)
      - GoogLeNet: Even deeper and introduces inception modules (groups of convolution layers)
      - ResNet (Residual Network): Even deeper and maintains performance via skip connections

## 145. Recurrent Neural Networks

- Breakdown of RNNs
  - Time series data - when you want to predict future behavior based on past behavior
      - Web logs, sensor logs, stock trades, etc
  - Data that consists of sequences of arbitrary length (instead of time)
      - Image captions, machine-generated music, etc
  - Feedback loop for back propagation (output of previous step becomes an input to the next step)
      - Can limit backpropagation to a limited number of time steps (truncated backpropagation through time)
  - RNN topologies
      - Sequence to sequence
        - Time series data, e.g. predict stock prices
      - Sequence to vector
        - Words in a sentence to sentiment analysis
      - Vector to sequence
        - Create captions from an image
      - Sequence to vector to sequence
        - Encoder -> Decoder
        - Machine translation
  - State dilution
      - This can be a problem from earlier time  steps that get diluted over time
      - Long short-term Memory (LSTM) cell
        - Maintains separate long-term states from short-term
      - Gated Recurrent Unit (GRU) cell
        - Simplified LSTM cell that performs equally well
  - Training RNN resource intensive
      - Very sensitive to topology, choice of hyperparameters
      - A wrong choice can lead to a RNN that doesn't converge at all

## 146. Tuning Neural Networks

- Breakdown of Tuning NNs
  - NNs are trained by gradient descent
  - Start at a random point and sample different solutions (weights) seeking to minimize some cost function, over many epochs
- Learning rate - indicates how far these samples are
  - A higher learning rate means you might overshoot the optimal solution
  - A lower learning rate will take too long to find the optimal solution
  - Learning rate is an example of hyperparameter
- Batch Size hyperparameter - indicates how many training samples are used within each batch of each epoch
  - Smaller batch size can work their way out of local minima more easily
  - Larger batch size can end up getting stuck in the wrong local minima (think of it as too heavy to move to another solution)

## 147. Regularization Techniques for NNs (Dropout, Early Stopping)

- Breakdown of NN Regularization Techniques
  - Any technique that prevent overfitting - models are good at making predictions on data they were trained on, but not on new data it hasn't seen before
  - Data can be segregated as training, evaluation or testing data sets
- Reduce number of layers, neurons reduces overall complexity of model
- Dropout removes some neurons at each epoch randomly
  - Forces weights to be spread out to other neurons
  - Prevents overfitting in each specific neurons
- Early stopping minimizes the number of epochs

## 148. L1 and L2 Regularization Term

- L1 and L2 Regularization Term
  - A regularization term is added as weights are learned
      - Prevent overfitting in general
  - L1 term = sum of the weights
      - Perform feature selection - entire features go to zero
      - Inefficient computationally (as more intensive)
      - Sparse output (as some features are removed)
  - L2 term = sum of the square of the weights
      - All features remain considered, just weighted
      - Efficient computationally
      - Dense output (not discarding features)
- Why would you choose L1 Regularization Term?
  - Feature selection can reduce dimensionality
      - Out of 100 features, only 10 may end up with non-zero cofficients
      - The sparse output can make up for its computational inefficiency (as more intensive)
      - But if you think all features are important, L2 is probably a better choice

## 149. The Vanishing Gradient Problem

- Breakdown of Vanishing Gradient Problem
  - When the slope of the learning curve approaches zero, things can get stuck
  - End up with very small numbers that slow down training or even introduce numerical errors (as slope becomes horizontal)
  - Problem with deeper networks and RNN's as these vanishing gradients propagate to deeper layers
  - Opposite problem is exploding gradient
- Fixing the Vanishing Gradient
  - Multi-level hierarachy training
      - Break up levels into their own sub-networks trained individually
  - Long Short-Term memory (LSTM) NN
      - A Recurrent NN that addresses vanishing gradient problem
  - Residual Network (ResNet) NN
      - Another RNN that addresses vanishing gradient problem
      - Ensemble of shorter networks
  - Choice of Activation Function
      - ReLU is a good choice
- Gradient Checking (debugging technique)
  - Numerically check the derivates computed during training
  - Useful for validating code of NN training

## 150. The Confusion Matrix

- Breakdown of the Confusion Matrix
  - Shows the true positives, true negative, false positives and false negatives
      Actual Yes	Actual No
    Predicted Yes	True Positive	False Positive
    Predicted No	False Negative	True Negative
  - Accuracy: Higher True Positive and Higher True Negative
  - Non-accurate: Higher False Positive and Higher False Negative
- Example from the AWS documentation
  - Flipped prediction as columns, and actual as rows
    True (below)	Romance	Thriller	Adventure	Total (True)	F1
    Romance	Tp			57.92%
(49.1k)	0.78
    Thriller		Tp		21.23%
(18.0k)	0.33
    Adventure			Tp	20.85%
(17.7k)	0.32
    Total (Predicted)	77.56%
(65.8k)	9.33%
(7910)	13.12%
(11.1k)	100.00%
(84.8k)	0.47
  - Number of correct / incorrect predictions per category
  - F1 scores per category
  - True category frequencies (Total column)
  - Predicted category frequencies (Total Row)

## 151. Precision, Recall, F1, AUC, and more

- Breakdown of Measuring Classification Models
  - Shows the true positives, true negative, false positives and false negatives

|  | Actual Yes | Actual No |  |
|--|------------|-----------|--|
| Predicted Yes | True Positive  | False Positive | Precision = Tp/(Tp+Fp)
| Predicted No  | False Negative | True Negative  |
|  | Recall = Tp/(Tp+Fn) | Specificity = Tn/(Tn+Fp) |

  - Recall = True Positive / (True Positive + False Negative)
      - True Positive Rate (aka Sensitivity) or Completeness
      - Percentage of positives rightly predicted
      - Good choice of metric when false negatives are important
        - Fraud detection
  - Precision = True Positive / (True Positive + False Positive)
      - Correct positives or accuracy of prediction
      - Percentage of relevant results
      - Good choice of metric when false positives are important
        - Medical screening
        - Drug testing
  - Specificity = True Negative / (True Negative + False Positive)
      - True Negative Rate
  - F1 = 2 x (Precision - Recall)/(Precision + Recall)
      - F1 = 2Tp / (2Tp + Fp + Fn)
        - F1 scores vary between 0 and 1, where 1 is perfect, and 0 is the worst
      - Harmonic mean of precision and sensitivity
      - Good choice of metric when precision AND recall are important
  - Accuracy = TP + TN / (TP + FP + FN + TN)
      - Ratio of the number of correctly classified items to the total number of (correctly and incorrectly) classified items
        - Ratio vary between zero (0) and one (1), with a 1 for perfect accuracy
      - Useful for classification models
      - Measures how close the predicted class values are to the actual values
  - [Balanced Accuracy][r01]
      - Metric that measures the ratio of accurate predictions to all predictions
      - Useful in classification models, where the number of positives or negatives differ greatly from each other in an imbalanced dataset, such as when only 1% of email is spam.
      - Defined as 0.5 * [(TP/P)+(TN/N)], with values ranging from 0 to 1
  - Mean Absolute Error (MAE)
      - Useful for regression problems where the goal is to predict continuous numeric values
      - Calculates the average absolute different between the predicted and actual values
      - Unlike MSE, MAE does not square the errors, so it is less sensitive to outliers
  - Root Mean Squared Error (RMSE)
      - Limitations: Suitable for predictions, but not classifications
      - Accuracy measurement by measuring the square root of the squared difference between predicted and actual values, and is averaged over all values
        - Values range from zero (0) to infinity, with smaller numbers indicating a better model fit to the data
      - Useful in regression analysis to understand model prediction error, and to indicate the presence of large errors or outliers
  - Receiver Operating Characteristic (ROC) Curve
      - Useful accuracy metric for [binary classification models][r04]
      - Plot of True Positive Rate (Recall) vs False Positive Rate
        - Various threshold settings
        - Points above the diagonal represent good classification
        - The more it is bent toward the upper-left the better
      - Area Under the Curve (AUC)
        - Useful to compare and evaluate classification models, where area under curve represents the probability of classifier is positive
        - ROC AUC of 0.5 is a random classifier, 1.0 is perfect
        - Good choice of metric for evaluating classification models, but not prediction models
  - Precision/Recall (P-R) Curve
      - Area Under the Curve (AUC)
        - Similar to ROC AUC as higher AUC better
        - The more it is bent toward the upper-right the better
        - Good choice for information retrieval problems
          ® Example searching large number of documents for a tiny number that is relevant

## 152. RMSE, R-squared, MAE

- Breakdown of Measuring Prediction Models
  - R-squared
      - Square of correlation coefficient between observed and predicted outcomes
  - Measuring error between actual and predicted values
      - Root Mean-Squared Error (RMSE)
      - Mean Absolute Error (MAE)

## 153. Ensemble Methods: Bagging and Boosting

- [Ensemble Methods][r02]
  - Use of multiple learning models and algorithms to gain more accurate predictions than any single individual learning algorithm
  - Proven to be efficient in diverse applications and learning settings
    - Cybersecurity and fraud detection
    - Remote sensing, predicting next steps in financial decision-making
    - Medical diagnosis
    - Computer vision and NLP
  - Different types of techniques used to train models, their composition, and the way they merge the different predictions into a single inference
  - Bagging
      - Limitations: Homogenous models
      - Random Sampling With Replacement
      - Each resampled model can be trained in parallel
      - Examples include Random Forest and Extra Trees
      - Random forest
          - Analogy: Monte Carlo simulation of decision trees
          - Decision trees are prone to overfitting
          - Make lots of decision trees and let them all vote the result
          - How do these decision trees differ?
  - Boosting
      - Limitations: Homogenous models
      - Each model is trained in sequence
        - Next model takes into account the previous' models results
        - Re-weight data and model before running next model
        - First model with equal weights
      - Examples include AdaBoost, Gradient Boosting, and XGBoost
        - XGBoost
            - Supervised algorithm that works well for detecting fraud with class imbalanced datasets
            - Datasets can be heterogeneous with missing values and imbalanced
            - XGBoost has built-in techniques for handling class imbalance, such as scale_pos_weight
            - Useful for regression, classification and ranking problems
        - Adaptive Boosting (AdaBoost)
            - Useful for correcting errors of weak classifiers, by giving more weight to incorrectly classified data during each iteration
              - Initially gives the same weight to each data set
              - Repeats the process until the difference between actual and predicted values fall below an acceptable threshold
            - Struggles with class imbalances and noisy data
        - Gradient Boosting
            - Uses the gradient of a loss function to improve model iteratively
            - Unlike AdaBoost, GB does not give incorrectly classified items more weight, but it optimizes the loss function
  - Stacking
      - Often uses heterogeneous models, where predictions of each individual estimator are stacked together and used as input to a final estimate that handles prediction
      - The final estimator's training process often use cross-validation
  - Averaging
      - Average results of each model's estimation
  - Majority voting
  - Bagging vs Boosting
    - Model Accuracy > Boosting > Bagging
    - Prevent Overfitting > Bagging > Boosting
    - Parallel Training > Bagging > Boosting
    - Boosting yields better accuracy, but bagging avoids overfitting
        - XGBoost is the latest method
    - Bagging can be trained in parallel, but boosting trains model sequentially

## 154. Automatic Model Tuning (AMT) in SageMaker

- Breakdown of SageMaker Automatic Model Tuning (AMT)
  - Issue of trial and error in hyperparameter tuning
  - Auto-tuning of hyperparameters
      - Define the hyperparameters that are important
      - Define the input ranges
      - Define the output metrics that you are optimizing for
  - SageMaker spins up a job that trains as many combinations as you'll allow
      - Training instances are spun up potentially a lot
  - Best feature is that It learns as it goes
      - It doesn't have to try every possible combination
  - The set of hyperparameters that produce the best results can then be deployed as a model
- Best practices for AMT
  - Don't optimize too many hyperparameters at once
  - Limit your ranges to a small range
  - Use logarithmic scales when appropriate
  - Don't run too many training instances concurrently
      - This limits how well the AMT can learn as it goes
      - AMT learns from previous iteration
      - Training too many in parallel will slow its learning
  - Make sure the correct objective metric is met at the end of each training instance

## 155. Hyperparameter Tuning in AMT

- Breakdown of AMT Auto-tuning of hyperparameters
  - Early Stopping
      - Stops a training job early if its not improving significantly
      - Reduces compute time, and avoids overfitting
      - Set to Auto-stopping
      - Depends on object metric after each epoch
  - Warm Start
      - Uses one or more previous tuning jobs as a starting point
      - Informs which hyperparameter combination to search next
      - Can be a way to start where you left off from a stopped hyperparameter job
      - Types of warm start
        - IDENTICAL_DATA_AND_ALGORITHM
        - TRANSFER_LEARNING
  - Resource Limit
      - Default limits for number of
        - Parallel tuning jobs
        - Hyperparameters
        - Training jobs per tuning job
      - Increasing this requires a quota increase from AWS Support
  - Strategy Type
      - Hyperband > Bayesian Optimization > Random Search > Grid Search
      - [Grid Search][r03]
        - Basic brute force approach to try every possible combination
        - Limited to categorical hyperparameters
        - Chooses combinations of values from the range of categorical values
      - Random Search
        - No dependence on previous jobs, so they can run in parallel
        - Chooses a random combination of hyperparameters on each job
      - Bayesian Optimization
        - Treats tuning as a regression problem
        - Learn from each run to converge on an optimal hyperparameter values
      - Hyperband
        - Appropriate for algorithms that publish results iteratively
          ® Like training a NN over several epochs
        - Dynamically allocate resources, e.g. early stopping, parallel, etc
        - Much faster than Bayesian Optimization or Random Search

## 156. SageMaker Autopilot / AutoML

- Breakdown of SageMaker Autopilot
  - Auto-select algorithm based on your data
      - Data processing
      - Model tuning
      - All infrastructure
  - Broadly this is called AutoML Wizard
      - It does all the trial and error for you
- Autopilot workflow
  - Load data from your S3 for training
  - Select your target column for prediction
  - Auto-model creation
      - Notebook is available for visibility and control
  - Auto-model leaderboard
      - Ranked list of recommended models
      - You can select one
  - Auto-deploy model
      - Monitor the model
      - Refine via notebook if needed
- Auto-training modes
  - Autopilot HPO if dataset > 100MB
      - Defaults to HPO if Autopilot cannot determine the size of your dataset
        - S3 bucket inside a private VPC
        - S3DataType is ManifestFile
        - S3Uri contains more than 1,000 items
  - Autopilot Ensembling if dataset < 100MB
  - Autopilot Hyperparameter Optimization (HPO)
      - Select algorithm relevant to your dataset
        - XGBoost, Linear learner, Deep Learning, etc
      - Select range of hyperparameters
        - Runs up to 100 trails to find optimal values
        - Bayesian optimization used if dataset < 100 MB
        - Multi-fidelity optimization used if dataset > 100MB
        - Early stopping if trial is performing poorly
  - Autopilot Ensembling
      - Runs 10 trials with different model and parameter settings
        - Models are combined with a stacking ensemble method
      - Trains several base models with AutoGluon library
        - Wider range of models, include tree-based and NN algorithms
- Autopilot intervention
  - Can add in human guidance
  - With or without code in SageMaker Studio or AWS SDK
- Autopilot Explainability
  - Integrates with SageMaker Clarify
      - Transparency on how models arrive at predictions
      - Feature attribution
        - Uses SHAP Baselines / Shapley Values
        - Based on research paper from cooperative game theory
        - Assigns each feature an importance value for a given prediction
- Problem types
  - Binary/ multi classification
  - Regression

## 157. SageMaker Studio, SageMaker Experiments

- Breakdown of New Features of SageMaker
  - SageMaker Studio IDE for ML
      - Combination of VS code and Jupyter Notebook
      - Similar to R Studio
      - No hardware to manage
  - SageMaker Experiments
      - Limitations
        - Lacks ML workflow orchestration or DAG visualization
        - Lacks tools for auditing and compliance
      - Organize, capture, compare and search your ML jobs or experiment trials

## 158. SageMaker Debugger

- Breakdown of SageMaker Debugger
  - Debugger dashboards
      - Saved internal model state at periodical intervals
      - Gradients / tensors over time as model is trained
      - Define rules for detecting unwanted conditions while training
        - A debug job is run for each rule you configure
      - Logs and triggers a CloudWatch event when a rule is hit
      - Built-in rules include
        - Monitor system bottlenecks
        - Profile model framework operations
        - Debug model parameters
  - Debugger ProfilerRule
      - ProfilerReport
      - Hardware system metrics
        - CPUBottlenck
        - GPUMemoryIncrease
      - Framework Metrics
        - StepOutlier
        - OverallFrameworkMetrics
        - MaxInitializationTime
  - Built-in actions to receive notifications
      - StopTraining(), Email()
      - In response to debugger rules
      - Sends notifications via SNS
  - Auto-generated training reports
  - Supported Frameworks and Algorithms
      - Tensorflow
      - PyTorch
      - MXNet
      - XGBoost
      - SageMaker generic estimator
        - Used with custom training containers
  - Debugger API available in GitHub
      - Construct hooks and rules
        - CreateTrainingJob
        - DescribeTrainingJob
      - SMDebug client library lets you register hooks for accessing training data

## 159. SageMaker Model Registry

- Breakdown of Model Registry
  - Similar to DockerHub but for models
  - Catalog models
  - Manage versions
  - Associate metadata
  - Manage approval status
  - Deploy models
  - Automate deployment with CI/CD
  - Share models
  - Integrate with SageMaker ModelCards
      - Web page about model (similar to product specification)

## 160. Analyzing Training Jobs with TensorBoard

- Breakdown of TensorBoard (Third-party tool)
  - Visualization toolkit for Tensorflow or PyTorch
      - Visualize loss and accuracy
      - Visualize model graph
      - View histograms of weight, biases over time
      - Project embeddings to lower dimensions
      - Profiling for optimization
  - Integrated with SageMaker console or via URL
      - Requires modification to your script

## 161. SageMaker Training at Large Scale: Training Compiler, Warm Pools

- Breakdown of SageMaker Training Techniques
  - Training Compiler (to be deprecated)
      - Integrated with AWS Deep Learning Containers (DLCs)
        - Cannot bring your own container
        - DLCs are pre-made Docker images for Tensorflow, PyTorch, etc
      - Can accelerate training by up to 50%
      - Incompatible with distributed training libraries
      - Best practices to use GPU instances, enable debugging
  - Managed Warm Pools
      - Retain and re-use provisioned infrastructure
      - Useful to speed things up
      - KeepAlivePeriodInSeconds setting
      - Use a persistent cache to store data across your training jobs

## 162. SageMaker Checkpointing, Cluster Health Checks, Automatic Restarts

- Breakdown of SageMaker Features
  - Checkpointing
      - Creates snapshots during training
      - You can restart from these snapshots
      - Automatic sync with S3
      - Analyze the model at different points
  - Cluster Health Checks and Auto Restarts
      - Auto-checks when using ml.g or ml.p instance types
      - Replaces any faulty instances automatically
      - Runs GPU health checks
      - Auto-restart the job when an instances fail (after replacing the bad instances)

## 163. SageMaker Distributed Training Libraries and Distributed Data Parallelism

- Breakdown of Distributed Training Libraries
  - Job parallelism - multiple training jobs in parallel
  - Individual jobs can also be parallelized
      - Data parallelism - gradient descent computation in parallel
      - Model parallelism - parallel model (shard a model to multiple instances in parallel)
  - Distributed computation in gradient descent (MinML)
      - AllReduce collective
      - Implemented in SageMaker Distributed Data Parallelism Library
  - Distributed communication between nodes
      - AllGather collective
      - Offloads communication to CPU, frees up GPU
  - Allows Third-party Distributed Data Library
      - PyTorch DistributedDataParallel (DDP)
      - Torchrun
      - Mpirun
      - DeepSpeed (Microsoft)
      - Horovod
  - Not compatible with SageMaker Training Compiler

## 164. SageMaker Model Parallelism Library

- Breakdown of Model Parallelism Library for PyTorch only
  - Distributed model to overcome GPU memory limits
      - Optimization state sharding
        - Requires a stateful optimizer, e.g. adam, fp16 etc
        - Generally useful for >1B parameters
        - Optimization state is just its weights
      - Activation Checkpoint
        - Saves memory at the expense of computation
        - Reduces memory usage by clearing activations of certain layers
          ® Recompute these during a backward pass
        - Activation Offloading
          ® Swaps checkpointed activations to and from the CPU
  - Sharded Data Parallelism
      - Combines both parallel data and models
        - Optimizer states are sharded
        - Trainable parameters are also sharded


## 165. Elastic Fabric Adapter (EFA) and MiCS

- Breakdown of Elastic Fabric Adapter (EFA) and MiCS
  - Use for accelerating training
  - Network device attached to your instances
      - High performance bandwidth that is similar to on-prem instances
      - Requires Nvidia GPUs (Nvidia Collective Communication Library)
- MiCS
  - Minimize the Communication Scale (MiCS)
  - Use for distributed training of models with >1T parameters
      - Sharded Data Parallelism
      - EFAs
      - Larger instances

## Links

[r01]: https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-metrics-validation.html
[r02]: https://aws.amazon.com/blogs/machine-learning/efficiently-train-tune-and-deploy-custom-ensembles-using-amazon-sagemaker/
[r03]: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html
[r04]: https://docs.aws.amazon.com/machine-learning/latest/dg/binary-model-insights.html