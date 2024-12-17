This code implements the PPO reinforcement learning algorithm to 
perform algorithmic trading using crypto exchange market data. 
The input data consists of 2D snapshots of various market metrics, which have been 
encoded into 1D arrays using an LSTM-based encoder-decoder model. The encoded data 
achieves approximately 95% decoding accuracy, preserving the key features of the original data 
while reducing its dimensionality.

The goal of this code is to explore the use of reinforcement learning for 
making trading decisions based on the encoded market data. Sensitive information, 
such as the encoded dataset, simulation function and proprietary details, has been 
removed from this version for demonstration purposes.