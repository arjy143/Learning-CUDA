/* MLP Forward Pass
    
   MLPs are just matrix computations, like Y = f(XW + b):
    - Y is the output (batch size * output dimension)
    - f is the activation function, like sigmoid or ReLU
    - X is the input matrix
    - W is the weights matrix
    - b is the bias matrix
    
   There are 3 kernels needed here:
    - applying activation function
    - matmul for XW
    - addition for bias

   This is so much more efficient than my stupid java coursework implementation of an MLP
   although this doesn't do any backprop yet.
 */
