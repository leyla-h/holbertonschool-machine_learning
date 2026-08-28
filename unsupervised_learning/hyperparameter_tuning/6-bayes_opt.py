#!/usr/bin/env python3
"""Optimizes a Keras neural network's hyperparameters on the MNIST
dataset using Bayesian optimization (GPyOpt).
"""
import numpy as np
import GPyOpt
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt


(X_train, Y_train), (X_test, Y_test) = K.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255
X_test = X_test.reshape(-1, 784).astype('float32') / 255
Y_train = K.utils.to_categorical(Y_train, 10)
Y_test = K.utils.to_categorical(Y_test, 10)

# hold out a validation set from the training data
val_split = 10000
X_val, Y_val = X_train[:val_split], Y_train[:val_split]
X_train, Y_train = X_train[val_split:], Y_train[val_split:]

iteration = 0


def build_model(lr, units, dropout, l2_weight, layers):
    """Builds a fully-connected Keras classifier with the given
    hyperparameters
    """
    reg = K.regularizers.l2(l2_weight)
    model = K.Sequential()
    model.add(K.layers.Input(shape=(784,)))

    for _ in range(int(layers)):
        model.add(K.layers.Dense(int(units), activation='relu',
                                  kernel_regularizer=reg))
        model.add(K.layers.Dropout(dropout))

    model.add(K.layers.Dense(10, activation='softmax'))

    optimizer = K.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def objective(x):
    """Trains a model with a given set of hyperparameters and returns
    the value of the satisficing metric (validation loss) to minimize

    x is a 2D numpy array of shape (1, 5) containing:
        x[:, 0] - learning rate
        x[:, 1] - number of units per layer
        x[:, 2] - dropout rate
        x[:, 3] - L2 regularization weight
        x[:, 4] - number of hidden layers
        x[:, 5] - batch size
    """
    global iteration
    iteration += 1

    lr = float(x[:, 0])
    units = float(x[:, 1])
    dropout = float(x[:, 2])
    l2_weight = float(x[:, 3])
    layers = float(x[:, 4])
    batch_size = int(x[:, 5])

    model = build_model(lr, units, dropout, l2_weight, layers)

    filename = (
        'checkpoints/model_lr{:.5f}_units{}_dropout{:.2f}_'
        'l2{:.5f}_layers{}_batch{}.h5'.format(
            lr, int(units), dropout, l2_weight, int(layers), batch_size)
    )

    checkpoint = K.callbacks.ModelCheckpoint(
        filename, monitor='val_loss', save_best_only=True)
    early_stop = K.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=30,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0)

    best_val_loss = min(history.history['val_loss'])

    print('Iteration {}: lr={:.5f}, units={}, dropout={:.2f}, '
          'l2={:.5f}, layers={}, batch_size={} -> val_loss={:.4f}'.format(
              iteration, lr, int(units), dropout, l2_weight,
              int(layers), batch_size, best_val_loss))

    return best_val_loss


domain = [
    {'name': 'lr', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
    {'name': 'layers', 'type': 'discrete', 'domain': (1, 2, 3)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)},
]


def f(x):
    """Wraps objective for GPyOpt's expected f(x) shape"""
    return np.array([[objective(x)]])


optimizer = GPyOpt.methods.BayesianOptimization(
    f=f,
    domain=domain,
    acquisition_type='EI',
    maximize=False)

optimizer.run_optimization(max_iter=30)

plt.figure()
optimizer.plot_convergence()
plt.savefig('convergence.png')

best_x = optimizer.x_opt
best_y = optimizer.fx_opt

report = []
report.append('Bayesian Optimization Report')
report.append('=============================')
report.append('')
report.append('Best hyperparameters found:')
report.append('  learning_rate: {:.5f}'.format(best_x[0]))
report.append('  units: {}'.format(int(best_x[1])))
report.append('  dropout: {:.2f}'.format(best_x[2]))
report.append('  l2_weight: {:.5f}'.format(best_x[3]))
report.append('  layers: {}'.format(int(best_x[4])))
report.append('  batch_size: {}'.format(int(best_x[5])))
report.append('')
report.append('Best validation loss (satisficing metric): {:.4f}'.format(
    best_y[0]))
report.append('')
report.append('All evaluations:')
for i, (x_i, y_i) in enumerate(zip(optimizer.X, optimizer.Y)):
    report.append(
        '  Iteration {}: lr={:.5f}, units={}, dropout={:.2f}, '
        'l2={:.5f}, layers={}, batch_size={} -> val_loss={:.4f}'.format(
            i + 1, x_i[0], int(x_i[1]), x_i[2], x_i[3],
            int(x_i[4]), int(x_i[5]), y_i[0]))

with open('bayes_opt.txt', 'w') as f_out:
    f_out.write('\n'.join(report))
