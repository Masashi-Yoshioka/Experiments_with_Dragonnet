import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.metrics import binary_accuracy
from keras.models import Model
from keras.layers import Layer, Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Define a (non-modified) cost function
def binary_classification_loss(concat_true, concat_pred):
    
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002 # Avoid t_pred = 0 and t_pred = 1
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


# Define metrics
def treatment_accuracy(concat_true, concat_pred):
    
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    
    return binary_accuracy(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    
    epsilons = concat_pred[:, 3]
    
    return tf.abs(tf.reduce_mean(epsilons))


# Define a class for epsilon
class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        self.epsilon = self.add_weight(name = 'epsilon',
                                       shape = [1, 1],
                                       initializer = 'RandomNormal',
                                       trainable = True)
        super(EpsilonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


# Define a modified cost function for targeted regularization
def make_tarreg_loss(ratio = 1., dragonnet_loss = dragonnet_loss_binarycross):
    
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02 # Avoid t_pred = 0 and t_pred = 1

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        loss = vanilla_loss + ratio * targeted_regularization
        
        return loss

    return tarreg_ATE_unbounded_domain_loss


# Create a "Dragonnet"
def make_dragonnet(input_dim, reg_l2):
    
    t_l1 = 0.
    t_l2 = reg_l2
    inputs = Input(shape = (input_dim, ), name = 'input')

    # Representation
    x = Dense(units = 200, activation = 'elu', kernel_initializer = 'RandomNormal')(inputs)
    x = Dense(units = 200, activation = 'elu', kernel_initializer = 'RandomNormal')(x)
    x = Dense(units = 200, activation = 'elu', kernel_initializer = 'RandomNormal')(x)

    t_predictions = Dense(units = 1, activation = 'sigmoid')(x)

    # Hypothesis
    y0_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(x)

    # Second layer
    y0_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(y1_hidden)

    # Third layer
    y0_predictions = Dense(units = 1, activation = None, kernel_regularizer = regularizers.l2(reg_l2),
                           name = 'y0_predictions')(y0_hidden)
    y1_predictions = Dense(units = 1, activation = None, kernel_regularizer = regularizers.l2(reg_l2),
                           name = 'y1_predictions')(y1_hidden)
    
    # epsilon
    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name = 'epsilon')
    
    # Finalize
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs = inputs, outputs = concat_pred)

    return model


# As a comparison, TARNET is also defined
def make_tarnet(input_dim, reg_l2):
    
    inputs = Input(shape = (input_dim, ), name = 'input')

    # Representation
    x = Dense(units = 200, activation = 'elu', kernel_initializer = 'RandomNormal')(inputs)
    x = Dense(units = 200, activation = 'elu', kernel_initializer = 'RandomNormal')(x)
    x = Dense(units = 200, activation = 'elu', kernel_initializer = 'RandomNormal')(x)

    t_predictions = Dense(units = 1, activation = 'sigmoid')(inputs) # Note the difference from the Dragonnet!

    # Hypothesis
    y0_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(x)

    # Second layer
    y0_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units = 100, activation = 'elu', kernel_regularizer = regularizers.l2(reg_l2))(y1_hidden)

    # Third layer
    y0_predictions = Dense(units = 1, activation = None, kernel_regularizer = regularizers.l2(reg_l2),
                           name = 'y0_predictions')(y0_hidden)
    y1_predictions = Dense(units = 1, activation = None, kernel_regularizer = regularizers.l2(reg_l2),
                           name = 'y1_predictions')(y1_hidden)
    
    # epsilon
    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name = 'epsilon')
    
    # Finalize
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs = inputs, outputs = concat_pred)

    return model


# Function for output
def _split_output(yt_hat, t, y, y_scaler, x):
    
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    g = yt_hat[:, 2].reshape(-1, 1).copy()

    y = y_scaler.inverse_transform(y.copy())

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x}


# Fit Dragonnet and predict
def train_and_predict_dragons(t, y_unscaled, x, targeted_regularization = True,
                              knob_loss = dragonnet_loss_binarycross, ratio = 1., dragon = 'dragonnet',
                              val_split = 0.2, batch_size = 32, verbose = 0):
    
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    outputs = []

    if dragon == 'dragonnet':
        dragonnet = make_dragonnet(x.shape[1], 0.01)
    elif dragon == 'tarnet':
        dragonnet = make_tarnet(x.shape[1], 0.01)

    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio = ratio, dragonnet_loss = knob_loss)
    else:
        loss = knob_loss

    yt = np.concatenate([y, t], axis = 1)

    import time;
    start_time = time.time()

    # Need to change hyperparameters according to performance on the validation subset
    # Here I tuned hyperparameters by repeating computation until validation loss reached as close to zero as possible
    dragonnet.compile(
        optimizer = Adam(learning_rate = 1e-3),
        loss = loss, metrics = metrics)

    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor = 'val_loss', patience = 10, min_delta = 0., mode = 'min'),
        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 2, verbose = verbose, mode = 'min',
                          min_delta = 0., cooldown = 0, min_lr = 0.)
    ]

    dragonnet.fit(x, yt, callbacks = adam_callbacks,
                  validation_split = val_split,
                  epochs = 500,
                  batch_size = batch_size, verbose = verbose)

    yt_hat = dragonnet.predict(x)
    
    outputs = _split_output(yt_hat, t, y, y_scaler, x)
    
    return outputs


# Following Shi et al. (2019), observations with extremely high/low propensity scores are omitted
def truncate_by_g(attribute, g, level = 0.01):
    
    keep_these = np.logical_and(g >= level, g <= 1.-level)

    return attribute[keep_these]


def truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level = 0.01):
    
    orig_g = np.copy(g)

    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)

    return q_t0, q_t1, g, t, y


# Average treatment effect on treated
def att_estimate(q_t0, q_t1, g, t, y, truncate_level = 0.01):
    
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)
    
    ite_t = (q_t1 - q_t0)[t == 1]
    att = ite_t.mean()

    return att


# Average treatment effect
def ate_from_att(q_t0, q_t1, g, t, y, truncate_level = 0.01):

    prob_t = t.mean()

    att = att_estimate(q_t0, q_t1, g, t, y, truncate_level = truncate_level)
    atnott = att_estimate(q_t0, q_t1, 1. - g, 1 - t, y, truncate_level = truncate_level)
    
    # Recover ATE from ATET
    ate = att * prob_t + atnott * (1. - prob_t)

    return ate


# Functions to realize the above DGP

# Covariance matrix of X
def fn_generate_cov(dim, corr):
    
    acc  = []
    for i in range(dim):
        row = np.ones((1, dim)) * corr
        row[0][i] = 1
        acc.append(row)
    
    return np.concatenate(acc, axis = 0)


# Generate X
def fn_generate_multnorm(nobs, corr, nvar):

    mu = np.zeros(nvar)
    std = (np.abs(np.random.normal(loc = 1, scale = .5, size = (nvar, 1))))**(1/2)
    
    # Generate random normal distribution
    acc = []
    for i in range(nvar):
        acc.append(np.reshape(np.random.normal(mu[i], std[i], nobs),(nobs, -1)))
    
    normvars = np.concatenate(acc, axis = 1)

    cov = fn_generate_cov(nvar, corr)
    C = np.linalg.cholesky(cov)

    Y = np.transpose(np.dot(C, np.transpose(normvars)))

    return Y


# Generate T that depends on X
def fn_confounded_treatment(X, degree):
    
    X_poly = PolynomialFeatures(degree = degree).fit_transform(X)
    X_poly = StandardScaler().fit_transform(X_poly)
    
    n, p = X_poly.shape
    
    beta1 = np.random.normal(loc = 0.1, scale = 0.1, size = (p, 1))
    
    p = np.exp(X_poly @ beta1)/(1 + np.exp(X_poly @ beta1))
    T = (p > 0.5) * 1
    
    return T


# Generate X, T and Y
def fn_generate_data(psi, nobs, nvar, corr, degree):

    X = fn_generate_multnorm(nobs, corr, nvar)
    T = fn_confounded_treatment(X, degree)
    e = np.random.normal(loc = 0, scale = 1, size = (nobs, 1))
    
    X_poly = PolynomialFeatures(degree = degree).fit_transform(X)
    X_poly = StandardScaler().fit_transform(X_poly)
    
    n, p = X_poly.shape
    
    beta2 = np.random.normal(loc = 1, scale = 1, size = (p, 1))
    
    Y = psi * T + X_poly @ beta2 + e
    
    return Y, T, X


# Define function to get the estimate of ATE

# Naive estimator: Difference in conditional means
def fn_psihat_naive(Y, T, X):
    
    qc_t1 = Y[np.where(T == 1)[0], :]
    qc_t0 = Y[np.where(T == 0)[0], :]
    psihat = np.mean(qc_t1) - np.mean(qc_t0)
    
    return psihat


# Linear regression of Y on T and X
def fn_psihat_ols(Y, T, X):
    
    X_all = np.concatenate((T, X), axis = 1)
    X_all = sm.add_constant(X_all)
    model = sm.OLS(Y, X_all)
    results = model.fit()
    psihat = results.params[1]
    
    return psihat


# Estimate potential output using Random Forest regression
def fn_psihat_rf(Y, T, X):
    
    n, p = X.shape

    TX = np.concatenate((T, X), axis = 1)
    T1X = np.concatenate((np.ones([n, 1]), X), axis = 1)
    T0X = np.concatenate((np.zeros([n, 1]), X), axis = 1)

    ### Hyperparameters are tuned by cross validation
    ### according to one set of (Y, T, X)
    #
    # from sklearn.model_selection import GridSearchCV
    #
    # psi = 10; nvar = 5; nobs = 1000; corr = 0.5; degree = 5
    # Y, T, X = fn_generate_data(psi, nobs, nvar, corr, degree)
    #
    # param_grid = {'n_estimators':[100, 200, 300],
    #               'max_depth':[10, 20, 30, None],
    #               'max_features': ['auto', 'sqrt'],
    #               'min_samples_leaf': [1, 2, 4],
    #               'min_samples_split': [2, 5, 10]}
    #
    # rfr = GridSearchCV(RandomForestRegressor(), param_grid = param_grid, verbose = 0,
    #                    scoring = 'neg_mean_squared_error', cv = 5)
    # rfr.fit(TX, Y.ravel())
    # print(rfr.best_params_)
    
    rfr = RandomForestRegressor(n_estimators = 200, max_depth = 20, min_samples_split = 2,
                                min_samples_leaf = 1, max_features = 'sqrt')
    rfr.fit(TX, Y.ravel())
    
    q_t1 = np.array(rfr.predict(T1X), ndmin = 2).T
    q_t0 = np.array(rfr.predict(T0X), ndmin = 2).T
    psihat = np.mean(q_t1 - q_t0)
    
    return psihat


# Doubly robust estimator with Random Forest
def fn_psihat_dml(Y, T, X):
    
    n, p = X.shape
    
    ### Hyperparameters are tuned by cross validation
    ### according to one set of (Y, T, X)
    #
    # from sklearn.model_selection import GridSearchCV
    # 
    # psi = 10; nvar = 5; nobs = 1000; corr = 0.5; degree = 5
    # Y, T, X = fn_generate_data(psi, nobs, nvar, corr, degree)
    # 
    # param_grid = {'n_estimators':[100, 200, 300],
    #               'max_depth':[10, 20, 30, None],
    #               'max_features': ['auto', 'sqrt'],
    #               'min_samples_leaf': [1, 2, 4],
    #               'min_samples_split': [2, 5, 10]}
    # 
    # rfc = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, verbose = 0,
    #                    scoring = 'neg_log_loss', cv = 5)
    # rfc.fit(X, T.ravel())
    # print(rfc.best_params_)
    
    rfc = RandomForestClassifier(n_estimators= 100, max_depth = 20, min_samples_split = 2,
                                 min_samples_leaf = 1, max_features = 'sqrt')
    rfc.fit(X, T.ravel())

    g = np.array(rfc.predict_proba(X)[:, 1], ndmin = 2).T
    # Observations with extremely high/low propensity scores are bounded instead of being omitted
    # because many of the observations are dropped otherwise
    g = np.maximum(g, 0.01)
    g = np.minimum(g, 0.99)

    TX = np.concatenate((T, X), axis = 1)
    T1X = np.concatenate((np.ones([n, 1]), X), axis = 1)
    T0X = np.concatenate((np.zeros([n, 1]), X), axis = 1)
    
    # Same hyperparameters as for fn_psihat_rf()
    rfr = RandomForestRegressor(n_estimators = 200, max_depth = 20, min_samples_split = 2,
                                min_samples_leaf = 1, max_features = 'sqrt')
    rfr.fit(TX, Y.ravel())

    q_t1 = np.array(rfr.predict(T1X), ndmin = 2).T
    q_t0 = np.array(rfr.predict(T0X), ndmin = 2).T
    
    psihat = np.mean(q_t1 - q_t0 + (T/g)*(Y - q_t1) - ((1.-T)/(1.-g))*(Y - q_t0))
    
    return psihat


# Dragonnet (and TARNET, with and without targeted regularization)
def fn_psihat_dragonnet(Y, T, X, dragon = 'dragonnet', targeted_regularization = True):
    
    results = train_and_predict_dragons(T, Y, X, dragon = dragon,
                                        targeted_regularization = targeted_regularization)
    q_t0 = results['q_t0']
    q_t1 = results['q_t1']
    g = results['g']
    t = results['t']
    y = results['y']
    
    psihat = ate_from_att(q_t0, q_t1, g, t, y)
    
    return psihat


# Compute bias and RMSE
def fn_bias_rmse(psi, psihats):
    
    psihats = np.array(psihats)
    b = psihats - psi
    bias = np.mean(b)
    rmse = np.sqrt(np.mean(b**2))
    
    return bias, rmse


def fn_simulation(estimator, psi, nobs, nvar, corr, degree, R):
    '''
    Monte Carlo simulation for ATE estimates
    Draw histogram of the estimates and compute bias & RMSE
    
    'estimator' should be one of the following strings:
        'naive': Naive estimator
        'ols': Linear regression
        'rf': Random Forest regression
        'dml': Doubly robust estimator with Random Forest
        'treg_dragonnet': Dragonnet with targeted regularization
        'dragonnet': Dragonnet without targeted regularization
        'treg_tarnet': TARNET with targeted regularization
        'tarnet': TARNET without targeted regularization
    '''
        
    # Obtain the name of the estimator
    estimators = ['naive', 'ols', 'rf', 'dml', 'treg_dragonnet', 'dragonnet', 'treg_tarnet', 'tarnet']
    names = ['Naive Estimator', 'Linear Regression', 'Random Forest Regression',
             'Doubly Robust Estimator with Random Forest', 'Dragonnet with Targeted Regularization',
             'Dragonnet without Targeted Regularization', 'TARNET with Targeted Regularization',
             'TARNET without Targeted Regularization']
    name = names[estimators.index(estimator)]

    psihats = []
    
    for r in tqdm(range(R)):
        
        # Generate the artificial data
        Y, T, X = fn_generate_data(psi, nobs, nvar, corr, degree)
        
        # Estimate the ATE
        if estimator == 'naive':
            psihat = fn_psihat_naive(Y, T, X)
        
        elif estimator == 'ols':
            psihat = fn_psihat_ols(Y, T, X)
        
        elif estimator == 'rf':
            psihat = fn_psihat_rf(Y, T, X)
        
        elif estimator == 'dml':
            psihat = fn_psihat_dml(Y, T, X)
        
        elif estimator == 'treg_dragonnet':
            psihat = fn_psihat_dragonnet(Y, T, X, dragon = 'dragonnet', targeted_regularization = True)

        elif estimator == 'dragonnet':
            psihat = fn_psihat_dragonnet(Y, T, X, dragon = 'dragonnet', targeted_regularization = False)

        elif estimator == 'treg_tarnet':
            psihat = fn_psihat_dragonnet(Y, T, X, dragon = 'tarnet', targeted_regularization = True)
        
        elif estimator == 'tarnet':
            psihat = fn_psihat_dragonnet(Y, T, X, dragon = 'tarnet', targeted_regularization = False)
        
        else: raise TypeError('This estimator is not available')
        
        psihats += [psihat]
    
    # Compute bias & RMSE
    bias, rmse = fn_bias_rmse(psi, psihats)
    print(f'{name}   Bias: {bias}, RMSE: {rmse}')
    
    # Draw the histogram of psihats
    plt.hist(psihats, label = '$\hat{\\psi}$')
    plt.title(f'ATE Estimates of {name} ($R$ = {R})')
    plt.axvline(psi, color = '#990000', label = f'True $\\psi$ = {psi}')
    plt.legend()
    plt.show()