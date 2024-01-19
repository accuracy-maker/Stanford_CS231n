# Stanford_CS231n

My self-learning of CS231n

## Assignment1

I have finished the KNN part, here is the code link: [KNN](https://drive.google.com/file/d/1ZtpfPpl0CZ6OLZwZCoIXTwAr81ZsRfxC/view?usp=sharing)

Now add to the SVM and Softmax part: [Softmax](https://drive.google.com/file/d/1Y_O9Q-owA9B6xFr_rCubDk_LY4jJT8O2/view?usp=sharing), [SVM](https://drive.google.com/file/d/1CTd4WN_opQON1ZrYSB5_AQIf8lj6QVOw/view?usp=sharing), [two_layer_network](https://colab.research.google.com/drive/1pONo9zI6Ym2mtPEcIB3MmcpWGW_PjcRl#scrollTo=9063768c)

In the assignment 1, I have learned a lot of new knowledage and set a futther understanding of gradient of loss function. I was used to use `loss.backward()` and `optimizer.step` in Pytorch to compute the loss and update the parameters of a loss function `f`. I was never focusing on the math behind this before. At present, I have a very intutive thoughts on loss function and gradient.

### Gradient

Many people learning AI must know the stotistic gradient descent (SGD) algorithm and are familar with concepts such as derivatives and gradient. However, I was confused about gradient everytime when I start to think about the details in graident descent. Why the gradient is the steepest direction of the function space? Why not other directional vector. Now, I figure it out. As we all know that every space in the university can be modeled by axis (meta vectors) and their combiantion. In deep learning setting, we want compute the gradient of loss function $L(W) = F(W)$, we need to update the $W$ in it's dimention space.

$$
W = \begin{bmatrix} W_x \\ W_y \\ W_z \\ ... \end{bmatrix}
$$

where $W$ contains many subvetors. When we compute the graident of $L(W)$, we must compute all parameters,

$$
\nabla L(W) = \begin{bmatrix} \nabla L(W_x) \\ \nabla L(W_y) \\ \nabla L(W_z) \\ \nabla L(W_{...}) \end{bmatrix}
$$

where every $\nabla L(W_i)$ is a direction vetor of this parameters' space that has both direction and scalar value. Given them, the largest scaler value is $\sqrt{\sum_i \nabla L(W_i)^2}$ and the direction is the combination of all vetors. It's every easy to think when only 2 dimentions.

#### Compute the Gradient

There are two methods to compute the gradient of a function: a slow, approximate but easy way  `numerical gradient` and a fast exact but more error-prone way  `analytic gradient`.

#### Numerical Gradient

The method is very straight forward computed by the definition of derivative,

$$
\frac{df(x)}{dx} = lim_{h -> 0} \frac{f(x+h) - f(x)}{h}
$$

where corresponding python code:

```python
def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```

#### Analytical Gradient

This method is to compute the formula of derivative function and use the matrix to compute all gradient at once.

##### Case Study: SVM Loss Function & Softmax Loss Function

###### SVM Loss Function

`SVM` loss function:

$$
L = \frac{1}{N} \sum_{i=1}^N \sum_{j \neq y_i} max(0,S_j - S_{y_i} + 1)
$$

where assuming that we have $N$ training samples, $C$ classes and function `S` to mapping $X$ to scores through $s = W^TX$.

So, how do we compute the loss and gradient $\nabla L$?

Assuming that we are classify some images and we have some arguements:

- W: A numpy array of shape (D, C) containing weights.
- X: A numpy array of shape (N, D) containing a minibatch of data.
- y: A numpy array of shape (N,) containing training labels; y[i] = c means
  that X[i] has label c, where 0 <= c < C.
- reg: the regulation part hyperparameter

We can easily propose a navie method using explict loop

```python
def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]
                loss += margin

  
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W


    return loss, dW
```

By using the loop, we can step by step compute every loss just according to the formula but why `dW[:,j] += X[i];dW[:,y[i]] -= X[i]`.

Let's do some analysis.

We can rewrite the SVM loss function:

$$
L(W) = \frac{1}{N} \sum_{i=1}^N \sum_{j \neq y_i} max(0,W_j^TX_i - W_{y_i}^TX_i + 1)
$$

Therefore, when the `max` function equals zero, the derivative equals zero too and when `max` function doesn't equal 0 (margin > 0), the function will be $W_j^TX_i - W_{y_i}^TX_i + 1$

And we compute the $\nabla W$,

$$
\nabla W = \begin{bmatrix} \nabla W_j \\ \nabla W_{y_i} \end{bmatrix} = \begin{bmatrix} X_i \\ -X_i \end{bmatrix}
$$

It's clear why I code these lines:

```python
for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]
                loss += margin
```

This method is very native and computation expensive. We can easily think about the vectorized version of SVM loss but how ???

**How can we transfer the loop to matrix operation?**

We can get some hints from the naive method:

- we must use the matrix $X$
- derivative $dW$ has the same shape with $W$: [D,C]
- we should use $X$ to dot product something to produce $dW$

Now, we need to do some mathematics. Let's consider a singe sample $i$

$$
L_i(W) = \sum_{j \neq y_i} max(0,W_j^TX_i - W_{y_i}^TX_i + 1)
$$

let's think about what times should we compute the derivatives? Assuming we have 3 classes ($C = 3$)

- if $s[y_i] > S[j] + 1$, $L_i = 0$, not compute
- if $S[j_1] + 1 > S[y_i] > S[j_2] + 1$, $L_i = W_{j_2}^TX_i - W_{y_i}^TX_i + 1 $, $dW[j_2] = X_i, dW[y_i] = -X_i, dW[j_1] = 0$
- if $S[y_i] < S[j] + 1$, $L_i = \sum_{j \neq y_i} max(0,W_j^TX_i - W_{y_i}^TX_i + 1)$, $dW[j_2] = X_i, dW[y_i] = -2X_i, dW[j_1] = X_i$.

We can find that: the derivates seem a linear combination of $X_i$ and some weights

$$
\nabla W = X_i * \begin{bmatrix} 1 & -2 & 1 \end{bmatrix}
$$

why [1,-2, 1]? That's the times when margin is greater than 0 (the score of incorrect classes outperform the score of correct class).

Therefore, we find a solution:

```python
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # num_train = X.shape[0]
    # scores = X.dot(W) # scores shape: [N,C]
    # for i in range(num_train):
    #   scores[i] = scores[i] - scores[i,y[i]] + 1
    # margin = np.maximum(0,scores)
    # margin[:,y] = 0
    # loss = np.sum(margin)
    # loss /= num_train
    # loss += reg * np.sum(W * W) 

    num_train = X.shape[0]
    scores = X.dot(W)  # Compute scores
    # print(scores[np.arange(num_train), y].shape)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)  # Correct class scores
    margins = np.maximum(0, scores - correct_class_scores + 1)  # Compute margins
    margins[np.arange(num_train), y] = 0  # Do not consider correct class in loss
    loss = np.sum(margins) / num_train  # Compute loss
    loss += reg * np.sum(W * W)  # Add regularization


    # num_class = W.shape[1]
    binary = margins
    binary[margins > 0] = 1
    # count the times when other classes outperform the correct class
    row_sum = np.sum(binary,axis = 1)
    binary[np.arange(num_train),y] = -row_sum
    dW = np.dot(X.T,binary) 
    dW /= num_train
    dW += 2 * reg * W
    # pass


    return loss, dW
```

From the matrix operation, we can conclude that every item in training samples update the $dW$ once at each class dimention, the weights are the `binary` matrix.

###### Softmax Loss Function

the `Softmax Function` is

$$
P(y = k|X=x_i) = \frac{e^{s_k}}{\sum_{j=1}^C e^{s_j}}
$$

where the $k$ is the correct class of the sample and overall classes are $C$.

The loss function is

$$
L = \frac{1}{N} \sum_{i} -log(\frac{e^{s_k}}{\sum_{j=1}^C e^{s_j}})
$$

Yeah, mathematic again!!!

Similar to SVM function, $W$ contains two parts: $W_k$ and $W_j$

We still consider one single sample:

$$
L_i = -log(\frac{e^{s_k}}{\sum_{j=1}^C e^{s_j}}) \\ = -(loge^{s_j} - log \sum_j e^{s_j}) \\ = log \sum_j e^{s_j} - loge^{s_k}
$$

Therefore,

$$
\nabla L_i = \nabla log \sum_j e^{s_j} - \nabla loge^{s_k}
$$

$$
\nabla log \sum_j e^{s_j} = \frac{1}{\sum_j e^{s_j}} * \nabla \sum_j e^{s_j} \\ = \frac{1}{\sum_j e^{s_j}} \sum_j \nabla e^{s_j} \\ = \sum_j \frac{e^{s_j} * X_i}{\sum_j e^{s_j}}
$$

$$
\nabla loge^{s_k} = X_i
$$

There are some small details need to notice: k = j and k != j

When we compute the j == k that means we compute the $dW_k$ so $\nabla loge^{s_k} = X_i$ works and the final gradient equals $\sum_j \frac{e^{s_j} * X_i}{\sum_j e^{s_j}} - X_i$

$$
\sum_j \frac{e^{s_j} * X_i}{\sum_j e^{s_j}} - X_i = (P - 1) * X_i
$$

when j != k, we compute the $dW_j$ so $\nabla loge^{s_k} = X_i$ is wrong because $\nabla W_j$ has nothing to do with $W_k$.

Therefore,

$$
L_i = P * X_i = \sum_j \frac{e^{s_j} * X_i}{\sum_j e^{s_j}}
$$

```python
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

   
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
      scores = X[i].dot(W) # score shape: [1,C]
      correct_class_score = scores[y[i]]
      sum_prob = np.sum(np.exp(scores))
      prob = np.exp(scores[y[i]]) / sum_prob
      loss += -1 * np.log(prob)
      for j in range(num_class):
        if j == y[i]:
          dW[:,j] += np.exp(correct_class_score) * X[i] / sum_prob - X[i]
        else:
          dW[:,j] += np.exp(scores[j]) / sum_prob * X[i]
  
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W) 
    dW += 2 * reg * W 
    # pass


    return loss, dW
```

Similar to SVM, we can use matrix operation to accelerate the speed. We can learn from the chain of thought of SVM and find that there are some weights multipling the $X_i$, which is [P-1, P , P] (if there are 3 classes).

Therefore, it's easily to implement this approach:

```python
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

  
    num_train = X.shape[0]
    num_class = W.shape[1]
  
    scores = X.dot(W) # scores.shape [N,C]
    exp_scores = np.exp(scores) # exp_scores.shape [N,C]
    column_sum = np.sum(exp_scores,axis = 1).reshape(-1,1) # column_sum.shape [N,1]
    prob_matrix = exp_scores / column_sum
    loss = -1 * np.log(exp_scores[np.arange(num_train),y] / column_sum)

    prob_matrix[np.arange(num_train),y] -= 1
    dW = np.dot(X.T,prob_matrix)

    dW /= num_train
    loss = np.mean(loss)
  
    loss += reg * np.sum(W*W)
    dW += 2 * reg * W
    # pass

    return loss, dW


```

So, we can conclude that:

- think of how many kind of parameters we should consider
- must use the $X_i$ to dot product something about score function

### Backpropagation

#### Jacobian Matrix

[Jacobian Wiki](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)

## Reference

- https://cs231n.github.io/optimization-1/
- https://cs231n.github.io/linear-classify/
- http://vision.stanford.edu/teaching/cs231n/handouts/linear-backprop.pdf
- http://cs231n.github.io/optimization-2
