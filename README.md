This project aims to collect, refine and experiment on recent machine learning models with their datasets and apply them in our researches. We currently make following contributions:

1. The APIs of collected models, either based on Tensorflow, PyTorch, or Keras, will be unified. With unified model APIs, common experiment routines can be written to avoid repeated coding for pre-processing, logging, visualization, etc., and make the experiment code more maintainable.

2. Every collected model will be broken into reusable pieces, if possible, so that they can be readily plugged into other models.

3. Every model will at least have one script with synthetic data to demo they are working. Models ever applied in our researches will have actual data and experiment code.

4. We'll try to comment the code as much as possible for both learning and documenting purpose.

Currently included or will be included soon:

1. Graph Convolutional Network (Tensorflow), adpated from https://github.com/tkipf/gcn.
    - Code restructured and improved, 5x faster on the DBLP dataset.
    - GCNLayer can now be plugged into other models.    
2. Deep-Q Learner (PyTorch & Keras), applied in the course project https://github.com/XinliYu/RL-Projects/tree/master/LunarLander.
    - Implements the Q-learning logic and capable of taking any neural network as the implementation of the Q-function.
3. DeepWalk/node2vec