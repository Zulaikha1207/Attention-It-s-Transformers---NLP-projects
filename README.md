## [NLP Transformers with Python] (https://ibm-learning.udemy.com/course/nlp-with-transformers/learn/lecture/26466724#overview)

Transformer models are the de-facto standard in modern NLP. They have proven themselves as the most expressive, powerful models for language by a large margin, beating all major language-based benchmarks time and time again. 

This repo consists of projects using key NLP frameworks such as:

- HuggingFace's Transformers
- TensorFlow 2
- PyTorch
- spaCy
- NLTK
- Flair

Projects on NLP use-cases:

- Language classification/sentiment analysis
- Named entity recognition (NER)
- Question and Answering
- Similarity/comparative learning

The project1 is a sentimental analysis classifier using Transformers and TensforFlow. This projects runs through the standard steps required to build a NLP model, namely, :

- Data preprocessing: Getting data from the Kaggle API, transform data to preprare it for sentimental analysis and tokenization
- Tf input pipeline: Build dataset (shuffle, batch, split data) for tensorflow
- Modellng and training: Initialise BERT model and define the architecture (the inpout layers, mask layer, embedding layers, output layer shapes, max pooling, activation layer, etc), set up the optimizer, loss function and evaluation metric. Train and save the model
- Getting predictions: Load the trained model, tokenise test data and make predictions

The Named Entity Recognition (NER) folder is an introduction to NER using spaCy, pulling stock data using Reddit API, extracting entities (organizations) from data and more.