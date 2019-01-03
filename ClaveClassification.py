from __future__ import print_function

import numpy as np
import csv

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')

import pickle
import lasagne

import theano
import theano.tensor as T
import binary_net, Q2b_net, Q3b_net, Q4b_net
import DataLoader

from collections import OrderedDict

def train_network(NNtype,save_path,hiddenNum,binary=True):
    np.random.seed(1234)  # for reproducibility
    # BN parameters
    batch_size = 10
    print("batch_size = " + str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # MLP parameters
    num_units = hiddenNum
    print("num_units = " + str(num_units))
    n_hidden_layers = 1
    print("n_hidden_layers = " + str(n_hidden_layers))

    # Training parameters
    num_epochs = 1000
    print("num_epochs = " + str(num_epochs))

    # BinaryOut
    activation = NNtype.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = NNtype.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")

    # BinaryConnect
    print("binary = " + str(binary))
    stochastic = False
    print("stochastic = " + str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = " + str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot"  # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = " + str(W_LR_scale))

    # Decaying LR
    LR_start = .003
    print("LR_start = " + str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = " + str(LR_fin))
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    print("LR_decay = " + str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    print("save_path = " + str(save_path))

    shuffle_parts = 1
    print("shuffle_parts = " + str(shuffle_parts))

    print('Loading dataset...')

    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_set, valid_set, test_set = DataLoader.DataLoader(seedval=1)

    train_set.X = 2 * train_set.X - 1.
    valid_set.X = 2 * valid_set.X - 1.
    test_set.X = 2 * test_set.X - 1.

    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)

    # Onehot the targets
    train_set.y = np.float32(np.eye(2)[train_set.y])
    valid_set.y = np.float32(np.eye(2)[valid_set.y])
    test_set.y = np.float32(np.eye(2)[test_set.y])

    # for hinge loss
    train_set.y = 2 * train_set.y - 1.
    valid_set.y = 2 * valid_set.y - 1.
    test_set.y = 2 * test_set.y - 1.

    print('Building the MLP...')

    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs', dtype=theano.config.floatX)
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
        shape=(None, 16),
        input_var=input)

    for k in range(n_hidden_layers):
        mlp = NNtype.DenseLayer(
            mlp,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=num_units)

        # mlp = lasagne.layers.BatchNormLayer(
        #     mlp,
        #     epsilon=epsilon,
        #     alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
            mlp,
            nonlinearity=activation)

    mlp = NNtype.DenseLayer(
        mlp,
        binary=binary,
        stochastic=stochastic,
        H=H,
        W_LR_scale=W_LR_scale,
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=2)

    # mlp = lasagne.layers.BatchNormLayer(
    #     mlp,
    #     epsilon=epsilon,
    #     alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    if binary:

        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = NNtype.compute_grads(loss, mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = NNtype.clipping_scaling(updates, mlp)

        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(
            updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err], allow_input_downcast=True)

    print('Training...')

    return NNtype.train(
        train_fn, val_fn,
        mlp,
        batch_size,
        LR_start, LR_decay,
        num_epochs,
        train_set.X, train_set.y,
        valid_set.X, valid_set.y,
        test_set.X, test_set.y,
        save_path,
        shuffle_parts)

if __name__ == "__main__":
    [errN, weightsN] = train_network(NNtype=binary_net, save_path='NoQuantization.npz', hiddenNum=16, binary=False)
    [err1, weights1] = train_network(NNtype=binary_net, save_path='Binary.npz', hiddenNum=16)
    [err2, weights2] = train_network(NNtype=Q2b_net, save_path='Q2b_net.npz', hiddenNum=16)
    [err3, weights3] = train_network(NNtype=Q3b_net, save_path='Q3b_net.npz', hiddenNum=16)
    [err4, weights4] = train_network(NNtype=Q4b_net, save_path='Q4b_net.npz', hiddenNum=16)
    errs = np.hstack([np.array(errN), np.array(err1), np.array(err2), np.array(err3), np.array(err4)])
    np.savetxt('test.out', errs, delimiter=',')

    writer = csv.writer(open("weightsN.csv", "w"))
    for row in weightsN:
        writer.writerow(str(row))

    writer = csv.writer(open("weights1.csv", "w"))
    for row in weights1:
        writer.writerow(str(row))

    writer = csv.writer(open("weights2.csv", "w"))
    for row in weights2:
        writer.writerow(str(row))

    writer = csv.writer(open("weights3.csv", "w"))
    for row in weights3:
        writer.writerow(str(row))

    writer = csv.writer(open("weights4.csv", "w"))
    for row in weights4:
        writer.writerow(str(row))