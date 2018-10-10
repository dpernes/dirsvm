import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
import warnings
import dill
from copy import deepcopy

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print ix, grad[ix]
    it.iternext() # step to next dimension

  return grad

class Svm(BaseEstimator, ClassifierMixin):

  def __init__(self, model_function_forward, model_function_backward,
               model_params, C=1, is_reg={}, is_optim={}, optimizer='SGD',
               lr=1e-3, lr_decay=0.99, momentum=0.9, rho1=0.9, rho2=0.999,
               num_epochs=1, model_function_post_train_step=None):
    self.model_function_forward = model_function_forward
    self.model_function_backward = model_function_backward
    self.model_params = model_params
    self.C = C

    if not is_reg:
      self.is_reg = {}
      for param in self.model_params:
        self.is_reg[param] = False
    else:
      self.is_reg = is_reg

    if not is_optim:
      self.is_optim = {}
      for param in self.model_params:
        self.is_optim[param] = True
    else:
      self.is_optim = is_optim

    self.lr = lr
    self.optimizer = optimizer

    if self.optimizer == 'SGD':
      self.init_sgd_momentum(lr_decay, momentum)
    elif self.optimizer == 'Adam':
      self.init_adam(rho1, rho2)

    self.num_epochs = num_epochs

    if model_function_post_train_step is None:
      self.model_function_post_train_step = self.empty_function
    else:
      self.model_function_post_train_step = model_function_post_train_step

    self.classes = np.array([-1,1])

  def init_sgd_momentum(self, lr_decay=None, momentum=None):
    self.lr_iter_ = self.lr
    if lr_decay is not None:
      self.lr_decay = lr_decay
    if momentum is not None:
      self.momentum = momentum
    self.model_params_vel = {}
    for param in self.model_params:
      self.model_params_vel[param] = 0.0

  def init_adam(self, rho1=None, rho2=None):
    self.lr_iter_ = self.lr
    if rho1 is not None:
      self.rho1 = rho1
    if rho2 is not None:
      self.rho2 = rho2
    self.time = 1
    self.model_params_1st_moment = {}
    self.model_params_2nd_moment = {}
    for param in self.model_params:
      self.model_params_1st_moment[param] = np.zeros_like(self.model_params[param])
      self.model_params_2nd_moment[param] = np.zeros_like(self.model_params[param])

  def empty_function(self, arg):
    return arg

  def hinge_loss(self, x, y):
    N = y.shape[0]
    loss = np.sum(np.maximum(1-x*y, 0))/N
    dx = (-y.astype(float)/N)*(1-x*y > 0)

    return loss, dx

  def svm_loss(self, x, y=None):
    f, cache_f = self.model_function_forward(x, self.model_params)
    if y is None:
      return f

    loss, df = self.hinge_loss(f, y)
    grads = self.model_function_backward(df, cache_f)

    # add the (L2) regularization term to the gradients
    for param in self.model_params:
      if self.is_reg[param] and self.is_optim[param]:
        loss += 0.5*np.sum(self.model_params[param]**2)/self.C
        grads[param] += self.model_params[param]/self.C

    return loss, grads

  def predict(self, x):
    f, _ = self.model_function_forward(x, self.model_params)

    # encode the predictions using the same class labels passed to fit()
    y_pred = self.classes[1]*(f >= 0) + self.classes[0]*(f < 0)

    return y_pred

  def decision_function(self, x):
    f, _ = self.model_function_forward(x, self.model_params)

    return f

  def sgd_momentum(self, w, dw, vel):
    next_vel = self.momentum*vel - self.lr_iter_*dw
    next_w = w + next_vel

    return next_w, next_vel

  def adam(self, w, dw, first_moment, second_moment, delta=1e-8):
    next_first_moment = self.rho1*first_moment + (1-self.rho1)*dw
    next_second_moment = self.rho2*second_moment + (1-self.rho2)*dw**2

    correct_first_moment = next_first_moment/(1-self.rho1**self.time)
    correct_second_moment = next_second_moment/(1-self.rho2**self.time)

    upd_w = (-self.lr_iter_*correct_first_moment/
            (np.sqrt(correct_second_moment) + delta))
    next_w = w + upd_w

    self.time += 1

    return next_w, next_first_moment, next_second_moment

  def train_step(self, x, y):
    loss, grads = self.svm_loss(x, y)

    for param in self.model_params:
      if self.is_optim[param]:
        if self.optimizer == 'SGD':
          (self.model_params[param],
           self.model_params_vel[param]) = self.sgd_momentum(
                                             self.model_params[param],
                                             grads[param],
                                             self.model_params_vel[param])
        elif self.optimizer == 'Adam':
          (self.model_params[param],
           self.model_params_1st_moment[param],
           self.model_params_2nd_moment[param]) = (
              self.adam(
                self.model_params[param],
                grads[param],
                self.model_params_1st_moment[param],
                self.model_params_2nd_moment[param]))

    self.model_params = self.model_function_post_train_step(self.model_params)

    return loss

  def fit(self, x, y_enc, batch_size=None, init_optim=True, verbose=False,
          print_every_nit=100):
    # transform the two class labels in y in -1 and 1
    self.classes = np.unique(y_enc)
    y = np.ones_like(y_enc).astype(int)
    y[y_enc==self.classes[0]] = -1

    # get the total number of iterations, using
    # the training set size, the number of epochs and the batch size
    N = y.shape[0]
    if batch_size is None:
      batch_size = N
    Nit = self.num_epochs * N/batch_size

    # initialize the optimizer
    if init_optim:
      if self.optimizer == 'SGD':
        self.init_sgd_momentum()
      elif self.optimizer == 'Adam':
        self.init_adam()

    loss_hist = []
    min_loss = 1e9
    loss_prev_epoch = min_loss
    params_best = deepcopy(self.model_params)
    for it in xrange(Nit):
      if batch_size < N:
        # randomly choose batch_size samples from the training set
        batch_mask = np.random.choice(N, batch_size)
      else:
        # batch mode
        batch_mask = np.arange(N)

      x_batch = x[batch_mask,:]
      y_batch = y[batch_mask]

      params_prev = deepcopy(self.model_params)
      loss = self.train_step(x=x_batch, y=y_batch)
      loss_hist.append(loss)

      if verbose and (it%print_every_nit == 0):
        print 'it: %d, loss = %f' %(it, loss)

      # save the best parameters
      if loss < min_loss:
        params_best = deepcopy(params_prev)
        min_loss = loss

      # actions to perform every epoch
      if it%(N/batch_size) == 0:
        # if the loss did not decrease,
        # decay the learning rate
        if self.optimizer == 'SGD':
          if loss > loss_prev_epoch:
            self.lr_iter_ *= self.lr_decay

        loss_prev_epoch = loss

    # choose the best parameters
    self.model_params = deepcopy(params_best)

    return loss_hist


class AsyTriSvm(Svm):

  def __init__(self, is_directional, weight_scale=1e-3, C=1, lr=1e-1,
               optimizer='SGD', lr_decay=0.99, momentum=0.9,
               rho1=0.9, rho2=0.999, num_epochs=1):
    self.is_directional = is_directional
    self.weight_scale = weight_scale
    self.is_linear = [not i for i in self.is_directional]

    F = len(self.is_directional)
    D = np.sum(self.is_directional) # number of directional features
    L = F - D                       # number of linear features

    w0 = np.array([0.0])
    wd = self.weight_scale*np.random.randn(D)
    zeta = 0.5*np.ones(D)
    theta = 2*np.pi*np.random.random(D)
    wl = self.weight_scale*np.random.randn(L)

    params = {}
    is_reg = {}
    is_optim = {}

    params['w0'] = w0
    is_reg['w0'] = False
    is_optim['w0'] = True

    params['wd'] = wd
    is_reg['wd'] = True
    is_optim['wd'] = True

    params['zeta'] = zeta
    is_reg['zeta'] = False
    is_optim['zeta'] = True

    params['theta'] = theta
    is_reg['theta'] = False
    is_optim['theta'] = True

    params['wl'] = wl
    is_reg['wl'] = True
    is_optim['wl'] = True

    super(AsyTriSvm, self).__init__(
      model_function_forward=self.asy_triangle_forward,
      model_function_backward=self.asy_triangle_backward,
      model_function_post_train_step=self.asy_triangle_post_train_step,
      model_params=params, is_optim=is_optim, C=C, is_reg=is_reg,
      num_epochs=num_epochs, lr=lr, optimizer=optimizer, lr_decay=lr_decay,
      momentum=momentum, rho1=rho1, rho2=rho2)

  def asy_triangle_forward(self, x, model_params):
    w0 = model_params['w0']
    wd = model_params['wd']
    wl = model_params['wl']
    theta = model_params['theta']
    zeta  = model_params['zeta']

    assert np.all(zeta >= 0) and np.all(zeta <= 1), (
      'Parameter zeta must lie in [0, 1], got zeta=%f' %(zeta))
    assert len(self.is_directional) == x.shape[1], (
      'The length of is_directional (%d) must be equal to the dimension '
      'of x (%d)') %(len(self.is_directional), x.shape[1])

    x_dir = x[:,self.is_directional];
    x_lin = x[:,self.is_linear];
    if x_dir.ndim == 1:
      x_dir = x_dir.reshape(-1,1)
    if x_lin.ndim == 1:
      x_lin = x_lin.reshape(-1,1)

    f = (w0 - np.matmul(signal.sawtooth(x_dir + theta, width=1-zeta),wd)
         + np.matmul(x_lin, wl))
    cache = x_dir, x_lin, w0, wd, theta, zeta, wl

    return f, cache

  def asy_triangle_backward(self, dout, cache):
    x_dir, x_lin, w0, wd, theta, zeta, wl = cache
    N, _ = x_dir.shape

    # the function is minimum at x=a (a<0)
    a = -2*np.pi * zeta
    # the function is minimum again at x=a+2*pi
    b = a + 2*np.pi

    # normalize the argument of the triangle wave to be in [a, b]
    arg_tri = (x_dir + theta)%(2*np.pi)
    arg_tri += -2*np.pi*(arg_tri > b) + 2*np.pi*(arg_tri < a)

    # compute the slopes of the ascending and descending parts
    slope_asc = 1/(np.pi*zeta)
    slope_dsc = 1/(np.pi*(zeta-1))

    grads = {}
    if self.is_optim['w0']:
      grads['w0'] = 1.0*np.sum(dout, axis=0)
    if self.is_optim['wd']:
      grads['wd'] = np.matmul((-signal.sawtooth(arg_tri, width=1-zeta)).T,
                    dout)
    if self.is_optim['theta']:
      grads['theta'] = np.matmul((slope_dsc * (arg_tri > 0)
                       + slope_asc * (arg_tri <= 0) * wd).T, dout)
    if self.is_optim['zeta']:
      grads['zeta'] = np.matmul((-1/(np.pi*(zeta-1)**2) * arg_tri
                      * (arg_tri > 0) + -1/(np.pi*zeta**2) * arg_tri
                      * (arg_tri <= 0) * wd).T, dout)
    if self.is_optim['wl']:
      grads['wl'] = np.matmul(x_lin.T, dout)

    return grads

  def asy_triangle_post_train_step(self, model_params, tol=0.01):
    w0 = model_params['w0']
    wd = model_params['wd']
    theta = model_params['theta']
    zeta = model_params['zeta']
    wl = model_params['wl']

    if self.is_optim['zeta']:
      # project zeta to the admissible range [0+tol, 1-tol]
      zeta[zeta<tol] = tol
      zeta[zeta>1-tol] = 1-tol

    if self.is_optim['theta']:
      # wrap the angle theta to be in [0,2*pi)
      theta = theta%(2*np.pi)

    model_params_upd = {}
    model_params_upd['w0'] = w0
    model_params_upd['wd'] = wd
    model_params_upd['theta'] = theta
    model_params_upd['zeta'] = zeta
    model_params_upd['wl'] = wl

    return model_params_upd

  def normalize_data(self, x):
    # assumes the input data x is in the interval [0, 1)
    # and normalizes the directional features to be in the
    # interval [0, 2*pi)
    x_norm = np.copy(x)
    x_norm[:,self.is_directional] = 2*np.pi*x_norm[:,self.is_directional]

    return x_norm

  def predict(self, x):
    # normalize the directional features to be in the interval [0, 2*pi)
    x_norm = self.normalize_data(x)

    y_pred = super(AsyTriSvm, self).predict(x_norm)

    return y_pred

  def decision_function(self, x):
    # normalize the directional features to be in the interval [0, 2*pi)
    x_norm = self.normalize_data(x)

    f = super(AsyTriSvm, self).decision_function(x_norm)

    return f

  def fit(self, x, y_enc, batch_size=None, init_params=True, init_optim=True,
          verbose=False, print_every_nit=100):
    F = len(self.is_directional)
    D = np.sum(self.is_directional) # number of directional features
    L = F - D                       # number of linear features

    if init_params:
      self.model_params['w0'] = np.array([0.0])
      self.model_params['wd'] = self.weight_scale*np.random.randn(D)
      self.model_params['theta'] = 2*np.pi*np.random.random(D)
      self.model_params['zeta'] = 0.5*np.ones(D)
      self.model_params['wl'] = self.weight_scale*np.random.randn(L)

    # normalize the input data to be in the interval [0, 2*pi)
    x_norm = self.normalize_data(x)

    ret = super(AsyTriSvm, self).fit(x_norm, y_enc, batch_size=batch_size,
          init_optim=init_optim, verbose=verbose,
          print_every_nit=print_every_nit)

    return ret


#class AsyTriSvm_2(Svm):

#  def __init__(self, is_directional, weight_scale=1e-3, C=1, lr=1e-1,
#               optimizer='SGD', lr_decay=0.99, momentum=0.9,
#               rho1=0.9, rho2=0.999, num_epochs=1):
#    self.is_directional = is_directional
#    self.weight_scale = weight_scale
#    self.is_linear = [not i for i in self.is_directional]

#    F = len(self.is_directional)
#    D = np.sum(self.is_directional) # number of directional features
#    L = F - D                       # number of linear features

#    w0 = np.array([0.0])
#    w1 = 2/np.pi * np.ones(D)
#    w2 = 2/np.pi * np.ones(D)
#    theta = 2*np.pi*np.random.random(D)
#    wl = self.weight_scale*np.random.randn(L)

#    params = {}
#    is_reg = {}
#    is_optim = {}

#    params['w0'] = w0
#    is_reg['w0'] = False
#    is_optim['w0'] = True

#    params['w1'] = w1
#    is_reg['w1'] = True
#    is_optim['w1'] = True

#    params['w2'] = w2
#    is_reg['w2'] = True
#    is_optim['w2'] = True

#    params['theta'] = theta
#    is_reg['theta'] = False
#    is_optim['theta'] = True

#    params['wl'] = wl
#    is_reg['wl'] = True
#    is_optim['wl'] = True

#    super(AsyTriSvm_2, self).__init__(
#      model_function_forward=self.asy_triangle_forward,
#      model_function_backward=self.asy_triangle_backward,
#      model_function_post_train_step=self.asy_triangle_post_train_step,
#      model_params=params, is_optim=is_optim, C=C, is_reg=is_reg,
#      num_epochs=num_epochs, lr=lr, optimizer=optimizer, lr_decay=lr_decay,
#      momentum=momentum, rho1=rho1, rho2=rho2)

#  def asy_triangle_forward(self, x, model_params):
#    w0 = model_params['w0']
#    w1 = model_params['w1']
#    w2 = model_params['w2']
#    wl = model_params['wl']
#    theta = model_params['theta']

#    x_dir = x[:,self.is_directional];
#    x_lin = x[:,self.is_linear];
#    if x_dir.ndim == 1:
#      x_dir = x_dir.reshape(-1,1)
#    if x_lin.ndim == 1:
#      x_lin = x_lin.reshape(-1,1)

#    arg_tri = (x_dir + theta)%(2*np.pi)

#    f  = (w0 - np.matmul(arg_tri*(arg_tri <= ((w2*2*np.pi)/(w1+w2))), w1)
#          + np.matmul((arg_tri - 2*np.pi)*(arg_tri > ((w2*2*np.pi)/(w1+w2))),
#          w2))
#    f = f.reshape(-1) + np.matmul(x_lin, wl)

#    cache = arg_tri, x_lin, w0, w1, w2, wl

#    return f, cache

#  def asy_triangle_backward(self, dout, cache):
#    arg_tri, x_lin, w0, w1, w2, wl = cache

#    N, D = arg_tri.shape

#    grads = {}
#    thr = (w2*2*np.pi)/(w1+w2)
#    if self.is_optim['w0']:
#      grads['w0'] = 1.0*np.sum(dout, axis=0)
#    if self.is_optim['w1']:
#      dw1 = arg_tri*(arg_tri <= thr)
#      grads['w1'] = np.matmul(dw1.T, dout)
#    if self.is_optim['w2']:
#      dw2 = (arg_tri - 2*np.pi)*(arg_tri > thr)
#      grads['w2'] = np.matmul(dw2.T, dout)
#    if self.is_optim['theta']:
#      dtheta = (-w1*(arg_tri <= thr)
#               + w2*(arg_tri > thr))
#      grads['theta'] = np.matmul(dtheta.T, dout)
#    if self.is_optim['wl']:
#      grads['wl'] = np.matmul(x_lin.T, dout)

#    return grads

#  def asy_triangle_post_train_step(self, model_params, tol=0.01):
#    w0 = model_params['w0']
#    w1 = model_params['w1']
#    w2 = model_params['w2']
#    theta = model_params['theta']
#    wl = model_params['wl']

#    if self.is_optim['w1']:
#      # project w1 to the admissible range (w1 > 0)
#      w1[w1<tol] = tol
#    if self.is_optim['w2']:
#      # project w2 to the admissible range (w1 > 0)
#      w2[w2<tol] = tol

#    if self.is_optim['theta']:
#      # wrap the angle theta to be in [0,2*pi)
#      theta = theta%(2*np.pi)

#    model_params_upd = {}
#    model_params_upd['w0'] = w0
#    model_params_upd['w1'] = w1
#    model_params_upd['w2'] = w2
#    model_params_upd['theta'] = theta
#    model_params_upd['wl'] = wl

#    return model_params_upd

#  def normalize_data(self, x):
#    # assumes the input data x is in the interval [0, 1)
#    # and normalizes the directional features to be in the
#    # interval [0, 2*pi)
#    x_norm = np.copy(x)
#    x_norm[:,self.is_directional] = 2*np.pi*x_norm[:,self.is_directional]

#    return x_norm

#  def predict(self, x):
#    # normalize the directional features to be in the interval [0, 2*pi)
#    x_norm = self.normalize_data(x)

#    y_pred = super(AsyTriSvm_2, self).predict(x_norm)

#    return y_pred

#  def decision_function(self, x):
#    # normalize the directional features to be in the interval [0, 2*pi)
#    x_norm = self.normalize_data(x)

#    f = super(AsyTriSvm_2, self).decision_function(x_norm)

#    return f

#  def fit(self, x, y_enc, batch_size=None, verbose=False, print_every_nit=100):
#    F = len(self.is_directional)
#    D = np.sum(self.is_directional) # number of directional features
#    L = F - D                       # number of linear features

#    self.model_params['w0'] = np.array([1.0])
#    self.model_params['w1'] = 2/np.pi * np.ones(D)
#    self.model_params['w2'] = 2/np.pi * np.ones(D)
#    self.model_params['theta'] = 2*np.pi*np.random.random(D)
#    self.model_params['wl'] = self.weight_scale*np.random.randn(L)

#    # normalize the input data to be in the interval [0, 2*pi)
#    x_norm = self.normalize_data(x)

#    ret = super(AsyTriSvm_2, self).fit(x_norm, y_enc, batch_size=batch_size,
#          verbose=verbose, print_every_nit=print_every_nit)

#    return ret

class AsyTriSvm_2step(AsyTriSvm):
  def __init__(self, is_directional, weight_scale=1e-3, C=1, lr=1e-1,
               optimizer='SGD', lr_decay=0.99, momentum=0.9,
               rho1=0.9, rho2=0.999, num_epochs1=1, num_epochs2=1):

    self.num_epochs1 = num_epochs1
    self.num_epochs2 = num_epochs2

    super(AsyTriSvm_2step, self).__init__(is_directional,
                                            weight_scale=weight_scale, C=C,
                                            lr=lr, optimizer=optimizer,
                                            lr_decay=lr_decay,
                                            momentum=momentum, rho1=rho1,
                                            rho2=rho2)

  def fit(self, x, y_enc, batch_size=None, verbose=False, print_every_nit=100):

    # optimize every parameter except zeta
    self.is_optim['w0'] = True
    self.is_optim['wd'] = True
    self.is_optim['theta'] = True
    self.is_optim['zeta'] = False
    self.is_optim['wl'] = True
    self.num_epochs = self.num_epochs1
    ret1 = super(AsyTriSvm_2step, self).fit(x, y_enc, batch_size=batch_size,
                                              init_params=True, init_optim=True,
                                              verbose=verbose,
                                              print_every_nit=print_every_nit)

    # optimize zeta and finetune remaining parameters
    self.is_optim['w0'] = True
    self.is_optim['wd'] = True
    self.is_optim['theta'] = True
    self.is_optim['zeta'] = True
    self.is_optim['wl'] = True
    self.num_epochs = self.num_epochs2
    ret2 = super(AsyTriSvm_2step, self).fit(x, y_enc, batch_size=batch_size,
                                              init_params=False, init_optim=False,
                                              verbose=verbose,
                                              print_every_nit=print_every_nit)
    return (ret1, ret2)

class SymTriSvm(AsyTriSvm):
  def __init__(self, is_directional, weight_scale=1e-3, C=1, lr=1e-1,
               optimizer='SGD', lr_decay=0.99, momentum=0.9,
               rho1=0.9, rho2=0.999, num_epochs=1):
    super(SymTriSvm, self).__init__(is_directional=is_directional,
                                      weight_scale=weight_scale, C=C, lr=lr,
                                      optimizer=optimizer, lr_decay=lr_decay,
                                      momentum=momentum, rho1=rho1, rho2=rho2,
                                      num_epochs=num_epochs)

    # in the symmetric case, zeta is fixed at its initial value (0.5)
    self.is_optim['zeta'] = False


#class Asy_tri_ker_svm(Svm):
#  def __init__(self, is_directional, weight_scale=1e-3, C=1, lr=1e-1,
#               optimizer='SGD', lr_decay=0.99, momentum=0.9,
#               rho1=0.9, rho2=0.999, num_epochs=1):
#    self.is_directional = is_directional
#    self.weight_scale = weight_scale
#    self.is_linear = [not i for i in self.is_directional]

#    # initialize the support vectors with an empty array before training
#    self.x_sup = np.array([])

#    # initialize the kernel matrices with an empty array also
#    self.k1 = np.array([])
#    self.k2 = np.array([])

#    params = {}
#    is_reg = {}
#    is_optim = {}

#    params['w0'] = np.array([0.0])
#    is_reg['w0'] = False
#    is_optim['w0'] = True

#    # the dimension of the alpha's will depend on the size of the
#    # training set, so by now we initialize it with an empty array
#    params['alpha1'] = np.array([])
#    is_reg['alpha1'] = True
#    is_optim['alpha1'] = True

#    params['alpha2'] = np.array([])
#    is_reg['alpha2'] = True
#    is_optim['alpha2'] = True

#    super(Asy_tri_ker_svm, self).__init__(
#      model_function_forward=self.asy_ker_svm_forward,
#      model_function_backward=self.asy_ker_svm_backward,
#      model_params=params, is_optim=is_optim, C=C, is_reg=is_reg, lr=lr,
#      optimizer=optimizer, lr_decay=lr_decay, momentum=momentum, rho1=rho1,
#      rho2=rho2, num_epochs=num_epochs)

#  def normalize_data(self, x):
#    # assumes the input data x is in the interval [0, 1)
#    # and normalizes the directional features to be in the
#    # interval [0, 2*pi)
#    x_norm = np.copy(x)
#    x_norm[:,self.is_directional] = 2*np.pi*x_norm[:,self.is_directional]

#    return x_norm

#  def kernels(self, x, z):
#    # computes the two asymmetric kernels
#    N = x.shape[0]
#    M = z.shape[0]

#    x_dir = x[:,self.is_directional];
#    x_lin = x[:,self.is_linear];
#    if x_dir.ndim == 1:
#      x_dir = x_dir.reshape(-1,1)
#    if x_lin.ndim == 1:
#      x_lin = x_lin.reshape(-1,1)

#    z_dir = z[:,self.is_directional];
#    z_lin = z[:,self.is_linear];
#    if z_dir.ndim == 1:
#      z_dir = z_dir.reshape(-1,1)
#    if z_lin.ndim == 1:
#      z_lin = z_lin.reshape(-1,1)

#    # linear kernel
#    k_lin = np.matmul(x_lin, z_lin.T)

#    # normalize the argument of the triangle wave to be in [-pi, pi]
#    arg_tri = (x_dir.reshape(N,1,-1) - z_dir.reshape(1,M,-1))%(2*np.pi)
#    arg_tri += -2*np.pi*(arg_tri > np.pi)

#    gt1 = np.sum((1 - 2/np.pi * arg_tri)*(arg_tri >= 0), axis=2)
#    gt2 = np.sum((1 + 2/np.pi * arg_tri)*(arg_tri <  0), axis=2)

#    k1 = gt1 + 0.5*k_lin
#    k2 = gt2 + 0.5*k_lin

#    return k1, k2

#  def asy_ker_svm_forward(self, x, model_params):
#    w0 = model_params['w0']
#    alpha1 = model_params['alpha1']
#    alpha2 = model_params['alpha2']

#    f = w0 + np.matmul(self.k1, alpha1) + np.matmul(self.k2, alpha2)
#    cache = None

#    return f, cache

#  def asy_ker_svm_backward(self, dout, cache):
#    grads = {}
#    grads['w0'] = 1.0*np.sum(dout, axis=0)
#    grads['alpha1'] = np.matmul((self.k1).T, dout)
#    grads['alpha2'] = np.matmul((self.k2).T, dout)

#    return grads

#  def predict(self, x):
#    # normalize the directional features to be in the interval [0, 2*pi)
#    x_norm = self.normalize_data(x)

#    # compute the kernel matrices
#    self.k1, self.k2 = self.kernels(x_norm, self.x_sup)

#    y_pred = super(Asy_tri_ker_svm, self).predict(x_norm)

#    return y_pred

#  def decision_function(self, x):
#    # normalize the directional features to be in the interval [0, 2*pi)
#    x_norm = self.normalize_data(x)

#    # compute the kernel matrices
#    self.k1, self.k2 = self.kernels(x_norm, self.x_sup)

#    f = super(Asy_tri_ker_svm, self).decision_function(x_norm)

#    return f

#  def fit(self, x, y_enc, batch_size=None, verbose=False, print_every_nit=100):
#    Nsup = x.shape[0]

#    # initialize the model parameters
#    self.model_params['w0'] = np.array([0.0])
#    self.model_params['alpha1'] = self.weight_scale*np.random.randn(Nsup)
#    self.model_params['alpha2'] = self.weight_scale*np.random.randn(Nsup)

#    # normalize the input data to be in the interval [0, 2*pi)
#    x_norm = self.normalize_data(x)

#    # the support vectors will be (a subset of) the training set
#    self.x_sup = self.normalize_data(x)

#    # compute the kernel matrices
#    self.k1, self.k2 = self.kernels(x_norm, self.x_sup)

#    ret = super(Asy_tri_ker_svm, self).fit(x_norm, y_enc,
#          batch_size=batch_size, verbose=verbose,
#          print_every_nit=print_every_nit)

#    return ret

def linear_kernel(x, y, *args, **kwargs):
    return np.dot(x, y.T)


def cosine_kernel(is_directional, x, y, *args, **kwargs):
    """
    num_dir = np.sum(is_directional)
    ret_dir = x[:, is_directional].reshape((x.shape[0], 1, num_dir)) - \
        y[:, is_directional].reshape((1, y.shape[0], num_dir))

    ret = np.dot(x[:, ~is_directional], y[:, ~is_directional].T) + \
        np.sum(np.cos(2.0 * np.pi * ret_dir), axis=2)
    """

    ret = np.zeros((x.shape[0], y.shape[0]))

    for i in xrange(x.shape[0]):
        next_ = x[i] * y
        next_[:, is_directional] = np.cos(2.0 * np.pi *
                                          (x[i] - y))[:, is_directional]
        ret[i, :] = np.sum(next_, axis=1)

    return ret


def triangular_kernel(is_directional, x, y, *args, **kwargs):
    def tri_wave(t):
        return signal.sawtooth(2.0 * np.pi * (t - 0.5), width=0.5)

    """
    num_dir = np.sum(is_directional)
    ret_dir = x[:, is_directional].reshape((x.shape[0], 1, num_dir)) - \
        y[:, is_directional].reshape((1, y.shape[0], num_dir))

    ret = np.dot(x[:, ~is_directional], y[:, ~is_directional].T) + \
        np.sum(tri_wave(ret_dir), axis=2)
    """

    ret = np.zeros((x.shape[0], y.shape[0]))

    for i in xrange(x.shape[0]):
        next_ = x[i] * y
        next_[:, is_directional] = tri_wave(x[i] - y)[:, is_directional]
        ret[i, :] = np.sum(next_, axis=1)

    return ret


class DirectionalKernelSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, is_directional, C=1., kernel='cosine', max_iter=-1):
        self.is_directional = is_directional

        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter

    def fit(self, X, y):
        self.is_directional = np.array(self.is_directional).astype(np.bool)
        if self.kernel == 'linear':
            kernel = 'linear'
        elif self.kernel == 'cosine':
            kernel = lambda x, y: cosine_kernel(self.is_directional,
                                                x, y)
        elif self.kernel == 'triangular':
            kernel = lambda x, y: triangular_kernel(self.is_directional,
                                                    x, y)

        self.svc_ = svm.SVC(C=self.C, kernel=kernel,
                            max_iter=self.max_iter, random_state=42)
        self.svc_.fit(X, y)

        return self

    def predict(self, X):
        if self.svc_ is None:
            return np.zeros(X.shape[0])

        return self.svc_.predict(X)

    def decision_function(self, X):
        if self.svc_ is None:
            return np.zeros(X.shape[0])
        return self.svc_.decision_function(X)


class AsyTriKernelSVM(Svm):
  def __init__(self, is_directional, weight_scale=1e-3, C=1, lr=1e-1,
               optimizer='SGD', lr_decay=0.99, momentum=0.9, rho1=0.9,
               rho2=0.999, max_iter=-1, num_epochs=1):
    self.is_directional = is_directional
    self.is_linear = [not i for i in self.is_directional]
    self.C = C
    self.max_iter = max_iter
    self.strisvm = DirectionalKernelSVM(is_directional, C=C,
                                        max_iter=self.max_iter,
                                        kernel='triangular')
    params = {}
    is_reg = {}
    is_optim = {}

    params['w0'] = np.array([0.0])
    is_reg['w0'] = False
    is_optim['w0'] = True

    # the dimension of the alpha's is not known beforehand
    # so by now we initialize it with an empty array
    params['alpha1'] = np.array([])
    is_reg['alpha1'] = False
    is_optim['alpha1'] = True

    params['alpha2'] = np.array([])
    is_reg['alpha2'] = False
    is_optim['alpha2'] = True

    self.support_vectors_ = np.array([])
    self.k1_ = np.array([])
    self.k2_ = np.array([])

    super(AsyTriKernelSVM, self).__init__(
      model_function_forward=self.asy_ker_svm_forward,
      model_function_backward=self.asy_ker_svm_backward,
      model_params=params, is_optim=is_optim, C=C, is_reg=is_reg, lr=lr,
      optimizer=optimizer, lr_decay=lr_decay, momentum=momentum, rho1=rho1,
      rho2=rho2, num_epochs=num_epochs)

  def normalize_data(self, x):
    # assumes the input data x is in the interval [0, 1)
    # and normalizes the directional features to be in the
    # interval [0, 2*pi)
    x_norm = np.copy(x)
    x_norm[:,self.is_directional] = 2*np.pi*x_norm[:,self.is_directional]

    return x_norm

  def kernels(self, x, z):
    # computes the two asymmetric kernels
    N = x.shape[0]
    M = z.shape[0]

    x_dir = x[:,self.is_directional];
    x_lin = x[:,self.is_linear];
    if x_dir.ndim == 1:
      x_dir = x_dir.reshape(-1,1)
    if x_lin.ndim == 1:
      x_lin = x_lin.reshape(-1,1)

    z_dir = z[:,self.is_directional];
    z_lin = z[:,self.is_linear];
    if z_dir.ndim == 1:
      z_dir = z_dir.reshape(-1,1)
    if z_lin.ndim == 1:
      z_lin = z_lin.reshape(-1,1)

    # linear kernel
    k_lin = np.matmul(x_lin, z_lin.T)

    # normalize the argument of the triangle wave to be in [-pi, pi]
    arg_tri = (x_dir.reshape(N,1,-1) - z_dir.reshape(1,M,-1))%(2*np.pi)
    arg_tri += -2*np.pi*(arg_tri > np.pi)

    gt1 = np.sum((1 - 2/np.pi * arg_tri)*(arg_tri >= 0), axis=2)
    gt2 = np.sum((1 + 2/np.pi * arg_tri)*(arg_tri <  0), axis=2)

    k1 = gt1 + 0.5*k_lin
    k2 = gt2 + 0.5*k_lin

    return k1, k2

  def asy_ker_svm_forward(self, x, model_params):
    w0 = model_params['w0']
    alpha1 = model_params['alpha1']
    alpha2 = model_params['alpha2']

    f = w0 + np.matmul(self.k1_, alpha1) + np.matmul(self.k2_, alpha2)
    cache = None

    return f, cache

  def asy_ker_svm_backward(self, dout, cache):
    grads = {}
    grads['w0'] = 1.0*np.sum(dout, axis=0)
    grads['alpha1'] = np.matmul((self.k1_).T, dout)
    grads['alpha2'] = np.matmul((self.k2_).T, dout)

    return grads

  def fit(self, x, y_enc, batch_size=None, verbose=False, print_every_nit=100):
    self.strisvm.C = self.C
    self.strisvm.max_iter = self.max_iter

    # first stage: fitting using the symmetric kernel
    self.strisvm.fit(x, y_enc)

    # normalize the input data to be in the interval [0, 2*pi)
    x_norm = self.normalize_data(x)

    # get the support vectors and the parameters
    self.support_vectors_ = x_norm[self.strisvm.svc_.support_, :]
    self.model_params['alpha1'] = self.strisvm.svc_.dual_coef_[0,:]
    self.model_params['alpha2'] = self.strisvm.svc_.dual_coef_[0,:]
    self.model_params['w0'] = self.strisvm.svc_.intercept_

    # compute the kernel matrices
    self.k1_, self.k2_ = self.kernels(x_norm, self.support_vectors_)

    # second stage: finetune the parameters for the asymmetric version
    ret = super(AsyTriKernelSVM, self).fit(x, y_enc,
            batch_size=batch_size,
            init_optim=True,
            verbose=verbose,
            print_every_nit=print_every_nit)

    return ret

  def predict(self, x):
    # normalize the directional features to be in the interval [0, 2*pi)
    x_norm = self.normalize_data(x)

    # compute the kernel matrices
    self.k1_, self.k2_ = self.kernels(x_norm, self.support_vectors_)

    y_pred = super(AsyTriKernelSVM, self).predict(x_norm)

    return y_pred

  def decision_function(self, x):
    # normalize the directional features to be in the interval [0, 2*pi)
    x_norm = self.normalize_data(x)

    # compute the kernel matrices
    self.k1_, self.k2_ = self.kernels(x_norm, self.support_vectors_)

    f = super(AsyTriKernelSVM, self).decision_function(x_norm)

    return f

