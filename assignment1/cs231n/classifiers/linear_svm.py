import numpy as np
from random import shuffle


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
    利用梯度下降算法分别对 w_y_i 和 w_j 求导后的结果，得到梯度
    同时对正则项中的 w 求导
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
                loss += margin
                # 如果 margin > 0 对 w_y_i 求导
                dW[:, y[i]] += -X[i]
                # 如果 margin > 0 对 w_j 求导
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # 对梯度取平均
    dW /= num_train

    # Add regularization to the loss.（添加正则项 loss，reg 相当于正则项的超参数）
    loss += reg * np.sum(W * W)

    # 添加正则项的导数
    dW += 2 * reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = np.dot(X, W)
    y_score = scores[np.arange(num_train), y].reshape((-1, 1))
    margins = np.maximum(0, scores - y_score + 1)  # broadcasting
    margins[np.arange(num_train), y] = 0  # 使 label 的分类损失由 1 变成 0
    loss = np.sum(margins, axis=1)
    loss = np.sum(loss) / num_train
    loss += reg * np.sum(W * W)

    # 另一种实现
    # y_score = scores[np.arange(num_train), y].reshape((-1, 1))
    # mask = scores - y_score + 1 > 0
    # scores = (scores - y_score + 1) * mask
    # loss = (np.sum(scores) - num_train * 1) / num_train
    # loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    mask = margins > 0
    ds = np.ones_like(scores)
    ds *= mask
    ds[np.arange(num_train), y] = -1 * (np.sum(ds, axis=1))  # 计算 j 不等于 y_i 的有效次数（损失函数大于 0）加和，用于对 w_y_i 求导，
    dW = np.dot(X.T, ds) / num_train  # 利用矩阵乘法，对 w_j 和 w_y_i 求导
    dW += reg * W  # 正则项求导

    ## 另一种实现
    # mask = scores - y_score + 1 > 0
    # ds = np.ones_like(scores)  # 初始化ds
    # ds *= mask  # 有效的score梯度为1，无效的为0
    # ds[np.arange(num_train), y] = -1 * (
    #         np.sum(mask, axis=1) - 1)  # 每个样本对应label的梯度计算了(有效的score次)，取负号；因为 mask 中正确分类 label 值也等于 1，所以要减去 1.
    # dW = np.dot(X.T, ds) / num_train  # 平均
    # dW += 2 * reg * W  # 加上正则项的梯度

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
