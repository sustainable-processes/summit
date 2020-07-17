from .base import Strategy, Transform
from summit.domain import Domain
from summit.utils.dataset import DataSet

import os

import numpy as np
import pandas as pd

import chemopt

import tensorflow as tf
import numpy as np
import logging
import json

import os.path as osp

from chemopt.logger import get_handlers
from collections import namedtuple

class DRO(Strategy):
    ''' Deep Reaction Optimizer from the paper:
    "Optimizing Chemical Reactions with Deep Reinforcement Learning"
    published by Zhenpeng Zhou, Xiaocheng Li, Richard N. Zare.

    Copyright (c) 2017 Zhenpeng Zhou

    Code is adapted from: https://github.com/lightingghost/chemopt

    Please cite their work, if you use this strategy.

    Parameters
    ----------
    domain: `summit.domain.Domain`
        A summit domain object
    transform: `summit.strategies.base.Transform`, optional
        A transform class (i.e, not the object itself). By default
        no transformation will be done the input variables or
        objectives.
    save_dir: string, optional
        Name of subfolder where temporary files during Gryffin execution are stored, i.e., summit/strategies/tmp_files/dro/<save_dir>.
        By default: None (i.e. no subfolder created, files stored in summit/strategies/tmp_files/dro)
    pretrained_model: string, optional
        Path to a pretrained DRO model (note that the number of inputs parameters should match the domain inputs)
        By default: a pretrained model (from chemopt/chemopt/save/) will be used

    Notes
    ----------

    Examples
    -------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import DRO
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[10, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='yield', description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    >>> strategy = DRO(domain)
    >>> next_experiments, xbest, fbest, param = strategy.suggest_experiments()
    >>> print(next_experiments)
    TODO: output


    '''

    def __init__(self, domain: Domain, transform: Transform=None, save_dir=None, pretrained_model=None, **kwargs):
        Strategy.__init__(self, domain, transform)

        # Create directories to store temporary files
        tmp_files = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp_files")
        tmp_dir = os.path.join(tmp_files, "dro")
        if not os.path.isdir(tmp_files):
            os.mkdir(tmp_files)
        # if a directory was specified create subfolder for storing files
        if save_dir:
            tmp_dir = os.path.join(tmp_dir, save_dir)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        self.domain = domain
        self.pretrained_model_path = pretrained_model
        self.infer_model_path = tmp_dir


    def suggest_experiments(self, prev_res: DataSet=None, prev_param=None):
        """ Suggest experiments using Nelder-Mead Simplex method

        Parameters
        ----------
        prev_res: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments.
            If no data is passed, the Nelder-Mead optimization algorithm
            will be initialized and suggest initial experiments.
        prev_param: file.txt TODO: how to handle this?
            File with parameters of Nelder-Mead algorithm from previous
            iterations of a optimization problem.
            If no data is passed, the Nelder-Mead optimization algorithm
            will be initialized.

        Returns
        -------
        next_experiments: DataSet
            A `Dataset` object with the suggested experiments by Nelder-Mead Simplex algorithm
        xbest: list
            List with variable settings of experiment with best outcome
        fbest: float
            Objective value at xbest
        param: list
            List with parameters and prev_param of Nelder-Mead Simplex algorithm (required for next iteration)
        """

        # Extract dimension of input domain
        self.dim = self.domain.num_continuous_dimensions()

        # Get bounds of input variables
        bounds = []
        for v in self.domain.variables:
            if not v.is_objective:
                bounds.append(v.bounds)
        self.bounds = np.asarray(bounds, dtype=float)

        # Initialization
        x0 = []
        self.y0 = []

        # Get previous results
        if prev_res is not None:
            inputs, outputs = self.transform.transform_inputs_outputs(prev_res)
            # Set up maximization and minimization
            for v in self.domain.variables:
                if v.is_objective:
                    a, b = np.asarray(v.bounds, dtype=float)
                    y = outputs[v.name]
                    y = (y - a) / (b - a)
                    if v.maximize:
                        y = 1 - y
                    outputs[v.name] = y
            self.x0 = inputs.data_to_numpy()
            self.y0 = outputs.data_to_numpy()
        # If no prev_res are given but prev_param -> raise error
        elif prev_param is not None:
            raise ValueError('Parameter from previous optimization iteration are given but previous results are '
                             'missing!')
        else:
            self.x0 = None
            self.y0 = None

        # if no previous results are given initialize with empty lists
        #if not len(x0):
        #    x0 = np.array(x0).reshape(0, len(bounds))
        #    y0 = np.array(y0).reshape(0, 2)


        logging.basicConfig(level=logging.INFO, handlers=get_handlers())
        logger = logging.getLogger()


        class StepOptimizer:
            def __init__(self, cell, ndim, nsteps, ckpt_path, infer_model_path, logger, constraints, x, y):
                self.logger = logger
                self.cell = cell
                self.ndim = ndim
                self.nsteps = nsteps
                self.ckpt_path = ckpt_path
                self.infer_model_path = infer_model_path
                self.constraints = constraints
                self.init_state = self.cell.get_initial_state(1, tf.float32)
                self.results = self.build_graph()
                self.x, self.y = x, y

                self.saver = tf.train.Saver(tf.global_variables())

            def get_state_shapes(self):
                return [(s[0].get_shape().as_list(), s[1].get_shape().as_list())
                        for s in self.init_state]

            def step(self, sess, x, y, state):
                feed_dict = {'input_x:0':x, 'input_y:0':y}
                for i in range(len(self.init_state)):
                    feed_dict['state_l{0}_c:0'.format(i)] = state[i][0]
                    feed_dict['state_l{0}_h:0'.format(i)] = state[i][1]
                new_x, new_state = sess.run(self.results, feed_dict=feed_dict)
                return new_x, new_state

            def build_graph(self):
                x = tf.placeholder(tf.float32, shape=[1, self.ndim], name='input_x')
                y = tf.placeholder(tf.float32, shape=[1, 1], name='input_y')
                state = []
                for i in range(len(self.init_state)):
                    state.append((tf.placeholder(
                                      tf.float32, shape=self.init_state[i][0].get_shape(),
                                      name='state_l{0}_c'.format(i)),
                                  tf.placeholder(
                                      tf.float32, shape=self.init_state[i][1].get_shape(),
                                      name='state_l{0}_h'.format(i))))

                with tf.name_scope('opt_cell'):
                    new_x, new_state = self.cell(x, y, state)
                    if self.constraints:
                        new_x = tf.clip_by_value(new_x, 0.001, 0.999)
                return new_x, new_state

            def load(self, sess, model_path):
                try:
                    logger.info('Reading model parameters from {}.'.format(model_path))
                    print(model_path)
                    self.saver.restore(sess, model_path)
                except:
                    raise FileNotFoundError('No checkpoint available')

            def get_init(self):
                x = np.random.normal(loc=0.5, scale=0.2, size=(1, self.ndim))
                x = np.maximum(np.minimum(x, 0.9), 0.1)
                init_state = [(np.zeros(s[0]), np.zeros(s[1]))
                              for s in self.get_state_shapes()]
                return x, init_state

            def run(self, prev_res=None, prev_param=None):
                with tf.Session() as sess:
                    if prev_res is None:
                        x, state = self.get_init()
                        print(self.ckpt_path)
                        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
                        print(ckpt)
                        model_id = ckpt.model_checkpoint_path.split("/")[-1]
                        init_model = os.path.join(os.path.dirname(chemopt.__file__), 'save', str(self.ndim) + '_inputs_standard', model_id)
                        self.load(sess, init_model)
                        self.infer_model_path = os.path.join(self.infer_model_path, model_id)
                        self.saver.save(sess, self.infer_model_path)
                    else:
                        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
                        model_id = ckpt.model_checkpoint_path.split("/")[-1]
                        self.infer_model_path = os.path.join(self.infer_model_path, model_id)
                        self.load(sess, self.infer_model_path)
                        state = prev_param[0]
                        if not np.array_equal(self.x, prev_param[1]):
                            raise ValueError("Values for input variables do not match requested points: {} != {}".format(str(self.x), str(prev_param[1])))
                        x, state = self.step(sess, self.x, self.y, state)
                        self.saver.save(sess, self.infer_model_path)
                return x, state

        def x_convert(x):
            real_x = np.zeros([self.dim])
            for i in range(self.dim):
                a, b = self.bounds[i]
                real_x[i] = x[0,i] * (b - a) + a
            return real_x

        def main(num_input=3, prev_res=None, prev_param=None):
            module_path = os.path.dirname(chemopt.__file__)
            path = osp.join(module_path, 'config_' + str(num_input) + '_inputs_standard.json')
            config_file = open(path)
            config = json.load(config_file,
                               object_hook=lambda d:namedtuple('x', d.keys())(*d.values()))
            saved_model_path = osp.join(module_path, str(config.save_path))

            cell = chemopt.rnn.StochasticRNNCell(cell=chemopt.rnn.LSTM,
                                         kwargs={'hidden_size':config.hidden_size},
                                         nlayers=config.num_layers,
                                         reuse=config.reuse)
            optimizer = StepOptimizer(cell=cell, ndim=config.num_params,
                                      nsteps=config.num_steps,
                                      ckpt_path=saved_model_path, infer_model_path=self.infer_model_path, logger=logger,
                                      constraints=config.constraints, x=self.x0, y=self.y0)
            x, state = optimizer.run(prev_res=prev_res, prev_param=prev_param)

            prev_param = [state, x]

            next_experiments = {}
            real_x = x_convert(x)
            i_inp = 0
            for v in self.domain.variables:
                if not v.is_objective:
                    next_experiments[v.name] = [real_x[i_inp]]
                    i_inp += 1
            next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))
            next_experiments[('strategy', 'METADATA')] = ['DRO']

            xbest, fbest = 0, 0

            tf.reset_default_graph()

            if not self.y0:
                self.y0 = [float("inf")]

            return next_experiments, xbest, self.y0[0], prev_param

        return main(num_input=self.dim, prev_res=self.y0, prev_param=prev_param)


