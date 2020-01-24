import gurobipy as gpy
import math
import itertools as it
import numpy as np

import sklearn.svm as svm
import sklearn.kernel_ridge as svk

import tensorflow as tf

import tqdm

from possibilearn.kernel import GaussianKernel
from possibilearn.fuzzifiers import LinearFuzzifier

def chop(x, minimum, maximum, tolerance=1e-4):
    '''Chops a number when it is sufficiently close to the extreme of
   an enclosing interval.

Arguments:

- x: number to be possibily chopped
- minimum: left extreme of the interval containing x
- maximum: right extreme of the interval containing x
- tolerance: maximum distance in order to chop x

Returns: x if it is farther than tolerance by both minimum and maximum;
         minimum if x is closer than tolerance to minimum
         maximum if x is closer than tolerance to maximum

Throws:

- ValueError if minimum > maximum or if x does not belong to [minimum, maximum]

'''
    if minimum > maximum:
        raise ValueError('Chop: interval extremes not sorted')
    #if  x < minimum or x > maximum:
    #    raise ValueError('Chop: value not belonging to interval')

    if x - minimum < tolerance:
        x = minimum
    if maximum - x < tolerance:
        x = maximum
    return x

def get_argument_value(key, opt_args, default_args):
    return opt_args[key] if key in opt_args else default_args[key]



def solve_optimization_tensorflow(x, mu, c=1.0, k=GaussianKernel(),
                                  adjustment=0,
                                  opt_args={}):
    '''Builds and solves the constrained optimization problem on the basis
   of the fuzzy learning procedure using the TensorFlow API.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the trade-off in joint radius/error optimization
- k: kernel function to be used
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- opt_args: arguments for TensorFlow (currently nothing)

Returns: a lists containing the optimal values for the independent
         variables chis of the problem

Throws:

- ValueError if optimization fails

'''

    default_args = {'init': None,
                    'init_bound': 0.1,
                    'num_iter': 100,
                    'optimizer': tf.optimizers.Adam(learning_rate=1e-4),
                    'tracker': tqdm.trange}

    init = get_argument_value('init', opt_args, default_args)

    m = len(x)

    if init is None:
      chis = [tf.Variable(0.01, name='chi_{}'.format(i),
                          trainable=True, dtype=tf.float32)
              for i in range(m)]
    elif init == 'random':
      l = get_argument_value('init_bound', opt_args, default_args)
      chis = [tf.Variable(ch, name='chi_{}'.format(i),
                          trainable=True, dtype=tf.float32)
              for i, ch in  enumerate(np.random.uniform(-0.1, 0.1, m))]

    else:
      chis = [tf.Variable(ch, name='chi_{}'.format(i),
                          trainable=True, dtype=tf.float32)
              for i, ch in  enumerate(init)]

    gram = np.array([[k.compute(x1, x2) for x1 in x] for x2 in x])

    def obj():
      penal = 10
      kernels = tf.constant(gram, dtype='float32')

      v = tf.tensordot(tf.linalg.matvec(kernels, chis), chis, axes=1)
      v -= tf.tensordot(chis, [k.compute(x_i, x_i) for x_i in x], axes=1)

      if adjustment:
        v += adjustment * tf.tensordot(chis, chis, axes=1)
      
      v += penal * tf.math.maximum(0, 1 - sum(chis))
      v += penal * tf.math.maximum(0, sum(chis) - 1)

      if c < np.inf:
        for ch, m in zip(chis, mu):
          v += penal * tf.math.maximum(0, ch - c*m)
          v += penal * tf.math.maximum(0, c*(1-m) - ch)

      return v

    opt = get_argument_value('optimizer', opt_args, default_args)
    
    num_iter = get_argument_value('num_iter', opt_args, default_args)

    tracker = get_argument_value('tracker', opt_args, default_args)

    for i in tracker(opt_args['num_iter']):
      old_chis = np.array([ch.numpy() for ch in chis])
      opt.minimize(obj, var_list=chis)
      new_chis = np.array([ch.numpy() for ch in chis])


    return [ch.numpy() for ch in chis]


def solve_optimization_gurobi(x,
                              mu,
                              c=1.0,
                              k=GaussianKernel(),
                              adjustment=0,
                              opt_args={}):
    '''Builds and solves the constrained optimization problem on the basis
   of the fuzzy learning procedure using the gurobi API.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the trade-off in joint radius/error optimization
- k: kernel function to be used
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- opt_args: arguments for gurobi ('time_limit' is the time in seconds before
            stopping the optimization process)

Returns: a lists containing the optimal values for the independent
         variables chis of the problem

Throws:

- ValueError if optimization fails

'''
    default_args = {'time_limit': 10*60}
    m = len(x)

    model = gpy.Model('possibility-learn')
    model.setParam('OutputFlag', 0)
    time_limit = get_argument_value('time_limit', opt_args, default_args)
    model.setParam('TimeLimit', opt_args['time_limit'])

    for i in range(m):
        if c < np.inf:
            model.addVar(name='chi_%d' % i, lb=-c*(1-mu[i]), ub=c*mu[i],
                         vtype=gpy.GRB.CONTINUOUS)

        else:
            model.addVar(name='chi_%d' % i, vtype=gpy.GRB.CONTINUOUS)

    model.update()

    chis = model.getVars()

    obj = gpy.QuadExpr()

    for i, j in it.product(range(m), range(m)):
        obj.add(chis[i] * chis[j], k.compute(x[i], x[j]))

    for i in range(m):
        obj.add(-1 * chis[i] * k.compute(x[i], x[i]))

    if adjustment:
        for i in range(m):
            obj.add(adjustment * chis[i] * chis[i])

    model.setObjective(obj, gpy.GRB.MINIMIZE)

    constEqual = gpy.LinExpr()
    constEqual.add(sum(chis), 1.0)

    model.addConstr(constEqual, gpy.GRB.EQUAL, 1)

    model.optimize()


    if model.Status != gpy.GRB.OPTIMAL:
        raise ValueError('optimal solution not found!')

    return [ch.x for ch in chis]

def solve_optimization(x, mu, c=1.0, k=GaussianKernel(),
                       tolerance=1e-4,
                       adjustment=0,
                       solve_strategy=solve_optimization_gurobi,
                       solve_strategy_args={'time_limit': 10*60}):
    '''Builds and solves the constrained optimization problem on the basis
   of the fuzzy learning procedure.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the trade-off in joint radius/error optimization
- k: kernel function to be used
- tolerance: tolerance to be used in order to clamp the problem solution to
             interval extremes
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- solve_strategy: algorithm to be used in order to numerically solve the
                  optimization problem
- solve_strategy_args: optional parameters for the optimization algorithm

Returns: a lists containing the optimal values for the independent
         variables chis of the problem

Throws:

- ValueError if c is non-positive or if x and mu have different lengths

'''
    if c <= 0:
        raise ValueError('c should be positive')

    if len(x) != len(mu):
        raise ValueError('patterns and labels have different length')


    mu = np.array(mu)

    chis = solve_strategy(x, mu, c, k, adjustment,
                                       solve_strategy_args)

    chis_opt = [chop(ch, l, u, tolerance)
                for ch, l, u in zip(chis, -c*(1-np.array(mu)), c*np.array(mu))]

    return chis_opt
    #return chis


def possibility_learn(x, mu, c=1, k=GaussianKernel(), sample_generator=None,
                      adjustment=0, fuzzifier=LinearFuzzifier(), crisp=False,
                      solve_strategy=solve_optimization_gurobi,
                      solve_strategy_args={'time_limit': 10*60},
                      return_vars=False,
                      return_profile=False):
    '''Induces a fuzzy membership function.

Arguments:

- x: iterable of objects
- mu: membership degrees of objects in x
- c: constant managing the trade-off in joint radius/error optimization
- k: kernel function to be used
- sample_generator: function randomly generating a given number of objects
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- fuzzifier: function to be used in order to get membership values of
             points falling outside the crisp set
- crisp: flag for triggering standard one-class classification

Returns: (f, e) with f being a function associating to a generic object the
         inferred degree of membership, and e being an estimate of the error

'''

    mu_train = [1]*len(mu) if crisp else mu

    try:
        chis = solve_optimization(x, mu_train, c=c, k=k, adjustment=adjustment,
                                  solve_strategy=solve_strategy,
                                  solve_strategy_args=solve_strategy_args)
    except ValueError as e:
        return (None, np.inf)

    gram = np.array([[k.compute(x1, x2) for x1 in x] for x2 in x])
    fixed_term = np.array(chis).dot(gram.dot(chis))

    def estimated_square_distance_from_center(x_new):
        ret = k.compute(x_new, x_new) \
               - 2 * np.array([k.compute(x_i, x_new) for x_i in x]).dot(chis) \
               + fixed_term
        return ret

    chi_SV_index = [i for i in range(len(chis)) if -c*(1-mu_train[i]) < chis[i] < c*mu_train[i]]
    chi_SV_square_distance = [estimated_square_distance_from_center(x[i])
                                for i in chi_SV_index]

    if len(chi_SV_square_distance) == 0:
        return (None, np.inf)

    SV_square_distance = np.mean(chi_SV_square_distance)
    num_samples = 500

    sample = map(np.array, sample_generator(num_samples))

    result = fuzzifier.get_fuzzified_membership(SV_square_distance, sample,
                                     estimated_square_distance_from_center,
                                     return_profile=return_profile)

    if return_profile:
        estimated_membership, mu_profile = result
    else:
        estimated_membership = result

    train_err = np.mean([(estimated_membership(x_i) - mu_i)**2
                         for x_i, mu_i in zip(x, mu)])

    result = [estimated_membership, train_err]
    if return_vars:
        result.append(chis)
    if return_profile:
        result.append(mu_profile)

    return result

def log_progress(sequence, every=None, size=None):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = size / 200     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{index} / ?'.format(index=index)
                else:
                    progress.value = index
                    label.value = u'{index} / {size}'.format(
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = str(index or '?')
        if size and index >= size:
            box.close()


def split_indices(paired_axioms, mus, percentages, shuffle=True):
    '''Generate a holdout triple of training-validate-test set.

    Arguments:

    - paired_axioms: list of paired indices to axioms
    - mu: list of paired membership values
    - percentages: list of three percentages of data to be put in training, validation,
      and test set, respectively (error is thrown if such percentages do no sum up as 1)

    Returns: (values_train, values_validate, values_test,
              mu_train, mu_validate)
    where

    - values_train: list of indices for training
    - values_validate: list of indices for model selection
    - values_test: list of indices for testing
    - mu_train: list of membership values for training
    - mu_validate: list of membership values for model selection
    - mu_test: list of membership values for testing
    '''

    assert(sum(percentages) == 1.)
    train_perc, validate_perc, test_perc = percentages

    n = len(paired_axioms)
    indices = range(n)

    if shuffle:
        indices = np.random.permutation(indices)

    num_train = int(n * train_perc)
    num_validate = int(n * validate_perc)
    num_test = int(n * test_perc)

    values_train = [paired_axioms[i] for i in indices[:num_train]]
    values_validate = [paired_axioms[i] for i in indices[num_train:num_train+num_validate]]
    values_test = [paired_axioms[i] for i in indices[num_train+num_validate:]]

    mu_train = [mus[i] for i in indices[:num_train]]
    mu_validate = [mus[i] for i in indices[num_train:num_train+num_validate]]
    mu_test = [mus[i] for i in indices[num_train+num_validate:]]

    return (values_train, values_validate, values_test, mu_train, mu_validate, mu_test)


def flatten(pair_list):
    '''Flattens out a list of pairs

    Arguments:

    - pair_list: the list of pairs to be flattened out

    Returns: list with all elements of pairs flattened out
    '''

    return [item for pair in pair_list for item in pair]


def cross_validation(data, labels, folds):
    '''Compute cross validation training and test sets

    Arguments:

    - data: list of available objects
    - labels: list of available labels
    - folds: number of cross validation folds

    Returns: list of <folds> pairs (values_train, values_test,
                                    labels_train, labels_test)
    '''

    assert(len(data)==len(labels))

    n = len(data)/folds
    partitioned_val = [data[i:i+n] for i in range(0, len(data), n)]
    partitioned_lab = [labels[i: i+n] for i in range(0, len(labels), n)]

    if n*folds != len(data):
        exceeding_val = partitioned_val.pop()
        for i in range(len(exceeding_val)):
            partitioned_val[i].append(exceeding_val[i])

        exceeding_lab = partitioned_lab.pop()
        for i in range(len(exceeding_lab)):
            partitioned_lab[i].append(exceeding_lab[i])

    result = []
    for test_index in range(folds):
        values_test = partitioned_val[test_index]
        train_set = partitioned_val[:test_index] + \
                    partitioned_val[test_index+1:]
        values_train = [item for sublist in train_set for item in sublist]

        labels_test = partitioned_lab[test_index]
        train_lab_set = partitioned_lab[:test_index] + \
                        partitioned_lab[test_index+1:]
        labels_train = [item for sublist in train_lab_set for item in sublist]

        result.append((values_train, values_test, labels_train, labels_test))
    return result


def model_selection_cv(x, mu, folds, cs, ks, sample_generator, log=False,
                    adjustment=0,
                    fuzzifier=LinearFuzzifier):
    '''Performs a model selection based on grid search over values for C and
   kernels.

Arguments:

- x: iterable of objects
- mu: iterable of membership degrees for objects
- folds: number of folds for internal cross-validation
- cs: possible values for the C constant
- ks: possible values for kernel function
- log: boolean flag activating a bar showing the computation progress
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- fuzzifier: function to be used in order to get membership values of
             points falling outside the crisp set

Returns: a tuple (c, k, (f, e)) being c and k the optimal values for
         the C constant and the kernel function, f the induced optimal
         membership function, and e an estimate of the error.
'''

    iterator = it.product(cs, ks)
    if log:
        iterator = log_progress(iterator, every=1, size=len(cs)*len(ks))

    best_result = (None, None, (None, np.inf))
    min_err = np.inf

    for c, k in iterator:
        folded_data = cross_validation(x, mu, folds)
        membership_rmse_metrics = []
        fold_num = 0
        for (paired_x_train, paired_x_test, paired_mu_train, paired_mu_test) in folded_data:
            print('internal fold {} of {}'.format(fold_num, folds))
            x_train = flatten(paired_x_train)
            mu_train = flatten(paired_mu_train)

            result = (c, k, possibility_learn(x_train, mu_train, c, k,
                                          sample_generator=sample_generator,
                                          adjustment=adjustment,
                                          fuzzifier=fuzzifier))

            estimated_membership = result[-1][0]
            fold_num += 1
            if estimated_membership is None:
                print('optimization did not succeed')
                continue
            x_test = flatten(paired_x_test)
            mu_test = flatten(paired_mu_test)
            membership_square_err = [(estimated_membership(v) - m)**2
                                 for v, m in zip(x_test, mu_test)]
            membership_rmse = math.sqrt(sum(membership_square_err) / len(x_test))
            membership_rmse_metrics.append(membership_rmse)

        avg_rmse = np.average(membership_rmse_metrics)
        if avg_rmse < min_err:
            best_result = result
            min_err = avg_rmse

    return best_result


def model_selection_holdout(paired_x_train, paired_mu_train,
                            paired_x_validate, paired_mu_validate,
                            cs, ks, sample_generator, log=False,
                            adjustment=0,
                            fuzzifier=LinearFuzzifier,
                            verbose=False,
                            crisp=False):
    '''Performs a holdout model selection based on grid search over values for
C and kernels.

Arguments:

- paired_x_train: iterable of pairs of objects to be used for training
- paired_mu_train: iterable of pairs membership degrees for objects to be used
  for training
- paired_x_validate: iterable of paired objects to be used for model selection
- paired_mu_validate: iterable of paired membership degrees for objects to be
  used for model selection
- cs: possible values for the C constant
- ks: possible values for kernel function
- log: boolean flag activating a bar showing the computation progress
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- fuzzifier: function to be used in order to get membership values of
             points falling outside the crisp set
- verbose: flag for verbose output
- crisp: flag triggering one-class classification

Returns: a tuple (c, k, (f, e)) being c and k the optimal values for
         the C constant and the kernel function, f the induced optimal
         membership function, and e the error over validation set.
'''

    iterator = it.product(cs, ks)
    if log:
        iterator = log_progress(iterator, every=1, size=len(cs)*len(ks))

    best_result = (None, None, (None, np.inf))
    min_err = np.inf


    x_train = flatten(paired_x_train)
    mu_train = flatten(paired_mu_train)
    x_validate = flatten(paired_x_validate)
    mu_validate = flatten(paired_mu_validate)


    for c, k in iterator:


        result = (c, k, possibility_learn(x_train, mu_train, c, k,
                                          sample_generator=sample_generator,
                                          adjustment=adjustment,
                                          fuzzifier=fuzzifier,
                                          crisp=crisp))

        estimated_membership = result[-1][0]

        if estimated_membership is None:
            if verbose:
                print('for C={} optimization did not succeed'.format(c))
            continue

        membership_square_err = [(estimated_membership(v) - m)**2
                                 for v, m in zip(x_validate, mu_validate)]
        membership_rmse = math.sqrt(sum(membership_square_err) / len(x_validate))

        if membership_rmse < min_err:
            min_err = membership_rmse
            best_result = (result[0], result[1], (result[2][0], membership_rmse))

    return best_result

def model_selection_holdout_reg(paired_x_train, paired_mu_train,
                            paired_x_validate, paired_mu_validate,
                            cs, gram, log=False,
                            verbose=False,
                            type='eps-svr'):
    '''Performs a holdout model selection based on grid search over values for
C and kernels, using sv-regression instead than fuzzy learn.

Arguments:

- paired_x_train: iterable of pairs of objects to be used for training
- paired_mu_train: iterable of pairs membership degrees for objects to be used
  for training
- paired_x_validate: iterable of paired objects to be used for model selection
- paired_mu_validate: iterable of paired membership degrees for objects to be
  used for model selection
- cs: possible values for the C constant
- gram: gram matrix for precomputed kernel
- log: boolean flag activating a bar showing the computation progress
- verbose: flag for verbose output
- type: 'eps-svr' for epsilon-svr, 'ridge' for ridge regression

Returns: a tuple (c, k, (f, e)) being c and k the optimal values for
         the C constant and the kernel function, f the induced optimal
         membership function, and e the error over validation set.
'''

    if log:
        iterator = log_progress(cs, every=1, size=len(cs))

    best_result = (None, None, (None, np.inf))
    min_err = np.inf


    x_train = flatten(paired_x_train)
    mu_train = flatten(paired_mu_train)
    x_validate = flatten(paired_x_validate)
    mu_validate = flatten(paired_mu_validate)
    
    gram_train = [[gram[i, j] for i in x_train] for j in x_train]
    gram_validate = [[gram[i, j] for i in x_train] for j in x_validate]
    
    if type == 'eps-svr':
        iterator = it.product(cs, [100000, 10000, 1000, 100, 10, 1, .1, .01, .001]) ## tanto viene selezionato sempre il valore piu' alto
        #iterator = it.product(cs, [10000, 1000, 100, .01, .001])
    else:
        iterator = cs


    for element in iterator:
        
        if type == 'eps-svr':
            c, epsilon = element
            svr = svm.SVR(kernel='precomputed', C=c, epsilon=epsilon)
        else:
            c = element
            svr = svk.KernelRidge(kernel='precomputed', alpha=1.0/(2.0*c))

        model = svr.fit(gram_train, mu_train)
        mu_train_hat = model.predict(gram_train)
        train_err = math.sqrt(np.mean((mu_train_hat - mu_train)**2))
        
        if type == 'eps-svr':
            result = (c, epsilon, (model, train_err))
        else:
            result = (c, (model, train_err))

        estimated_membership = result[-1][0]

        if estimated_membership is None:
            if verbose:
                print('for C={} optimization did not succeed'.format(c))
            continue
        
        mu_validate_hat = model.predict(gram_validate)
        membership_rmse = math.sqrt(np.mean((mu_validate_hat - mu_validate)**2))

        if membership_rmse < min_err:
            min_err = membership_rmse
            if type == 'eps-svr':
                best_result = (result[0], result[1], (result[-1][0], membership_rmse))
            else:
                best_result = (result[0], (result[-1][0], membership_rmse))

    return best_result
