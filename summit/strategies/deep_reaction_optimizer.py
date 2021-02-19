from .base import Strategy, Transform
from summit.domain import *
from summit.utils.dataset import DataSet
from summit import get_summit_config_path


import numpy as np
import pandas as pd

import logging
import json
import os.path as osp
import os
import pathlib
import uuid
from collections import namedtuple
from copy import deepcopy


class DRO(Strategy):
    """Deep Reaction Optimizer (DRO)

    The DRO relies on a pretrained RL policy that can predict a next set of experiments
    given a set of past experiments. We suggest reading the notes below before using the DRO.

    Parameters
    ----------

    domain: :class:`~summit.domain.Domain`
        A summit domain object
    transform: :class:`~summit.strategies.base.Transform`, optional
        A transform class (i.e, not the object itself). By default
        no transformation will be done the input variables or
        objectives.
    pretrained_model_config_path: string, optional
        Path to the config file of a pretrained DRO model (note that the number of inputs parameters should match the domain inputs)
        By default: a pretrained model will be used.
    model_size: string, optional
        Whether the model (policy) has the same size as originally published by the developers of DRO ("standard"),
        or whether the model is bigger w.r.t. number of pretraining epochs, LSTM hidden size, unroll_length ("bigger").
        Note that the pretraining can increase exponentially when changing these hyperparameters and the number of input variables,
        the number of epochs the each bigger model was trained can be found in the "checkpoint" file in the respective
        `save directory <https://github.com/sustainable-processes/chemopt/tree/master/chemopt/save>`_.
        By default: "standard" (these models were all pretrained for 50 epochs)

    Attributes
    ----------
    xbest, internal state
        Best point from all iterations.
    fbest, internal state
        Objective value at best point from all iterations.
    param, internal state
        A dict containing: state of LSTM of DRO, last requested point, xbest, fbest,
        number of iterations (corresponding to the unroll length of the LSTM)

    Examples
    -------

    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import DRO
    >>> from summit.utils.dataset import DataSet
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> strategy = DRO(domain)

    Notes
    -------

    The DRO requires Tensorflow version 1, while all other parts of Summit use Tensorflow version 2. Therefore,
    we have created a Docker container for running DRO which has TFv1 installed. We also have an option in the pip
    package to install TFv1.

    However, if you simply want to analyse results from a DRO run (i.e., use from_dict), then you will not get a
    tensorflow import error.

    We have pretrained policies for domains with up to six continuous decision variables

    For applying the DRO it is necessary to define reasonable bounds of the objective variable, e.g., yield in [0, 1],
    since the DRO normalizes the objective function values to be between 0 and 1.

    The DRO is based on the paper in ACS Central Science by [Zhou]_.

    References
    ----------

    .. [Zhou] Z. Zhou et al., ACS Cent. Sci., 2017, 3, 1337–1344.
       DOI: `10.1021/acscentsci.7b00492 <https://doi.org/10.1021/acscentsci.7b00492>`_


    """

    def __init__(
        self,
        domain: Domain,
        transform: Transform = None,
        pretrained_model_config_path=None,
        model_size="standard",
        **kwargs
    ):
        Strategy.__init__(self, domain, transform)

        # Create directories to store temporary files
        summit_config_path = get_summit_config_path()
        self.uuid_val = uuid.uuid4()  # Unique identifier for this run
        tmp_dir = summit_config_path / "dro" / str(self.uuid_val)
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

        self._pretrained_model_config_path = pretrained_model_config_path
        self._infer_model_path = tmp_dir
        self._model_size = model_size
        self.prev_param = None

    def suggest_experiments(self, prev_res: DataSet = None, **kwargs):
        """Suggest experiments using the Deep Reaction Optimizer

        Parameters
        ----------
        num_experiments: int, optional
            The number of experiments (i.e., samples) to generate. Default is 1.
        prev_res: :class:`~summit.utils.data.DataSet`, optional
            Dataset with data from previous experiments.
            If no data is passed, the DRO optimization algorithm
            will be initialized and suggest initial experiments.

        Returns
        -------
        next_experiments : :class:`~summit.utils.data.DataSet`
            A Dataset object with the suggested experiments

        Notes
        -------


        """
        import tensorflow as tf

        if tf.__version__ != "1.13.1":
            raise ImportError(
                """Tensorflow version 1.13.1 needed for DRO, which is different than the versions
                needed for other strategies. We suggest using the docker container marcosfelt/summit:dro.
                """
            )

        # Extract dimension of input domain
        self.dim = self.domain.num_continuous_dimensions()

        # Get bounds of input variables
        bounds = []
        for v in self.domain.input_variables:
            if isinstance(v, ContinuousVariable):
                bounds.append(v.bounds)
            elif isinstance(v, CategoricalVariable):
                if v.ds is not None:
                    descriptor_names = v.ds.data_columns
                    descriptors = np.asarray(
                        [v.ds.loc[:, [l]].values.tolist() for l in v.ds.data_columns]
                    )
                else:
                    raise ValueError("No descriptors given for {}".format(v.name))
                for d in descriptors:
                    bounds.append([np.min(np.asarray(d)), np.max(np.asarray(d))])

        # Get bounds of objective
        obj_maximize = False
        obj_bounds = None
        for v in self.domain.output_variables:
            if obj_bounds is not None:
                raise ValueError(
                    "DRO can not handle multiple objectives. Please use transform."
                )
            obj_bounds = v.bounds
            if v.maximize:
                obj_maximize = True
        self.bounds = np.asarray(bounds, dtype=float)

        # Initialization
        self.x0 = None
        self.y0 = None
        # Get previous results
        if prev_res is not None:
            inputs, outputs = self.transform.transform_inputs_outputs(
                prev_res, categorical_method="descriptors"
            )
            # Set up maximization and minimization and normalize inputs (x) and outputs (y)
            for v in self.domain.variables:
                if v.is_objective:
                    a, b = np.asarray(v.bounds, dtype=float)
                    y = outputs[v.name]
                    y = (y - a) / (b - a)
                    if v.maximize:
                        y = 1 - y
                    outputs[v.name] = y
                else:
                    a, b = np.asarray(v.bounds, dtype=float)
                    x = inputs[v.name]
                    x = (x - a) / (b - a)
                    inputs[v.name] = x
            self.x0 = inputs.data_to_numpy()
            self.y0 = outputs.data_to_numpy()
        # If no prev_res are given but prev_param -> raise error
        elif self.prev_param is not None:
            raise ValueError(
                "Parameter from previous optimization iteration are given but previous results are "
                "missing!"
            )

        # TODO: how does DRO handle constraints?
        if self.domain.constraints != []:
            raise NotImplementedError("DRO can not handle constraints yet.")

        next_experiments, param = self.main(
            num_input=self.dim, prev_res=[self.x0, self.y0], prev_param=self.prev_param
        )

        objective_dir = -1.0 if obj_maximize else 1.0
        self.fbest = (
            objective_dir * param["fbest"] * (obj_bounds[1] - obj_bounds[0])
            + obj_bounds[0]
        )
        self.prev_param = param

        # Do any necessary transformations back
        next_experiments = self.transform.un_transform(
            next_experiments, categorical_method="descriptors"
        )

        return next_experiments

    def reset(self):
        """Reset internal parameters"""
        self.prev_param = None

    def to_dict(self):
        """Convert hyperparameters and internal state to a dictionary"""
        if self.prev_param is not None:
            params = deepcopy(self.prev_param)
            tup_to_json = [
                list(e) for e in [list(t) for t in list([params["state"]])][0]
            ]
            params["state"] = [
                [tup_to_json[0][0].tolist(), tup_to_json[0][1].tolist()],
                [tup_to_json[1][0].tolist(), tup_to_json[1][1].tolist()],
            ]
            params["xbest"] = params["xbest"].tolist()
            params["fbest"] = params["fbest"].tolist()
            params["last_requested_point"] = params["last_requested_point"].tolist()
        else:
            params = None
        strategy_params = dict(
            prev_param=params,
            pretrained_model_config_path=self._pretrained_model_config_path,
            model_size=self._model_size,
        )
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
        dro = super().from_dict(d)
        params = d["strategy_params"]["prev_param"]
        if params is not None:
            params["state"] = tuple(
                [
                    tuple([np.array(s, dtype=np.float32) for s in e])
                    for e in params["state"]
                ]
            )
            params["xbest"] = np.array(params["xbest"])
            params["fbest"] = np.array(params["fbest"])
            params["last_requested_point"] = np.array(params["last_requested_point"])
        dro.prev_param = params
        return dro

    def x_convert(self, x):
        real_x = np.zeros([self.dim])
        for i in range(self.dim):
            a, b = self.bounds[i]
            real_x[i] = x[0, i] * (b - a) + a
        return real_x

    def main(self, num_input=3, prev_res=None, prev_param=None):
        import chemopt
        from chemopt.logger import get_handlers

        x0, y0 = prev_res[0], prev_res[1]
        module_path = os.path.dirname(chemopt.__file__)
        if self._pretrained_model_config_path:
            path = self._pretrained_model_config_path
        else:
            path = osp.join(
                module_path,
                "config_"
                + str(num_input)
                + "_inputs_"
                + str(self._model_size)
                + ".json",
            )
        config_file = open(path)
        config = json.load(
            config_file, object_hook=lambda d: namedtuple("x", d.keys())(*d.values())
        )
        saved_model_path = osp.join(
            os.path.dirname(os.path.realpath(path)), str(config.save_path)
        )
        if prev_param:
            if prev_param["iteration"] > config.unroll_length:
                raise ValueError(
                    "Number of iterations exceeds unroll length of the pretrained model!"
                )

        logging.basicConfig(level=logging.WARNING, handlers=get_handlers())
        logger = logging.getLogger()

        cell = chemopt.rnn.StochasticRNNCell(
            cell=chemopt.rnn.LSTM,
            kwargs={"hidden_size": config.hidden_size},
            nlayers=config.num_layers,
            reuse=config.reuse,
        )
        optimizer = self.StepOptimizer(
            cell=cell,
            ndim=config.num_params,
            nsteps=config.num_steps,
            ckpt_path=saved_model_path,
            infer_model_path=self._infer_model_path,
            logger=logger,
            constraints=True,
            x=x0,
            y=y0,
        )
        x, state = optimizer.run(prev_res=y0, prev_param=prev_param)

        real_x = self.x_convert(x)
        next_experiments = {}
        i_inp = 0
        for v in self.domain.variables:
            if not v.is_objective:
                next_experiments[v.name] = [real_x[i_inp]]
                i_inp += 1
        next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))
        next_experiments[("strategy", "METADATA")] = ["DRO"]

        param = {}
        if not y0:
            y0 = np.array([[float("inf")]])
            param["iteration"] = 0
        else:
            param["iteration"] = prev_param["iteration"] + 1
        if not prev_param:
            self.fbest = y0[0]
            self.xbest = real_x
        elif y0 < prev_param["fbest"]:
            self.fbest = y0[0]
            self.xbest = real_x
        else:
            self.fbest = prev_param["fbest"]
            self.xbest = prev_param["xbest"]

        param.update(
            {
                "state": state,
                "last_requested_point": x,
                "xbest": self.xbest,
                "fbest": self.fbest,
            }
        )

        tf.reset_default_graph()

        return next_experiments, param

    class StepOptimizer:
        def __init__(
            self,
            cell,
            ndim,
            nsteps,
            ckpt_path,
            infer_model_path,
            logger,
            constraints,
            x,
            y,
        ):
            self.logger = logger
            self.cell = cell
            self.ndim = ndim
            self.nsteps = nsteps
            self.ckpt_path = ckpt_path
            self._infer_model_path = infer_model_path
            self.constraints = constraints
            self.init_state = self.cell.get_initial_state(1, tf.float32)
            self.results = self.build_graph()
            self.x, self.y = x, y

            self.saver = tf.train.Saver(tf.global_variables())

        def get_state_shapes(self):
            return [
                (s[0].get_shape().as_list(), s[1].get_shape().as_list())
                for s in self.init_state
            ]

        def step(self, sess, x, y, state):
            feed_dict = {"input_x:0": x, "input_y:0": y}
            for i in range(len(self.init_state)):
                feed_dict["state_l{0}_c:0".format(i)] = state[i][0]
                feed_dict["state_l{0}_h:0".format(i)] = state[i][1]
            new_x, new_state = sess.run(self.results, feed_dict=feed_dict)
            return new_x, new_state

        def build_graph(self):
            x = tf.placeholder(tf.float32, shape=[1, self.ndim], name="input_x")
            y = tf.placeholder(tf.float32, shape=[1, 1], name="input_y")
            state = []
            for i in range(len(self.init_state)):
                state.append(
                    (
                        tf.placeholder(
                            tf.float32,
                            shape=self.init_state[i][0].get_shape(),
                            name="state_l{0}_c".format(i),
                        ),
                        tf.placeholder(
                            tf.float32,
                            shape=self.init_state[i][1].get_shape(),
                            name="state_l{0}_h".format(i),
                        ),
                    )
                )

            with tf.name_scope("opt_cell"):
                new_x, new_state = self.cell(x, y, state)
                if self.constraints:
                    new_x = tf.clip_by_value(new_x, 0.001, 0.999)
            return new_x, new_state

        def load(self, sess, model_path):
            try:
                self.saver.restore(sess, model_path)
            except:
                raise FileNotFoundError("No checkpoint available")

        def get_init(self):
            x = np.random.normal(loc=0.5, scale=0.2, size=(1, self.ndim))
            x = np.maximum(np.minimum(x, 0.9), 0.1)
            init_state = [
                (np.zeros(s[0]), np.zeros(s[1])) for s in self.get_state_shapes()
            ]
            return x, init_state

        def run(self, prev_res=None, prev_param=None):
            with tf.Session() as sess:
                if prev_res is None:
                    x, state = self.get_init()
                    ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
                    model_id = ckpt.model_checkpoint_path.split("/")[-1]
                    init_model = os.path.join(os.path.dirname(self.ckpt_path), model_id)
                    self.load(sess, init_model)
                    self._infer_model_path = os.path.join(
                        self._infer_model_path, model_id
                    )
                    self.saver.save(sess, self._infer_model_path)
                else:
                    ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
                    model_id = ckpt.model_checkpoint_path.split("/")[-1]
                    self._infer_model_path = os.path.join(
                        self._infer_model_path, model_id
                    )
                    self.load(sess, self._infer_model_path)
                    state = prev_param["state"]
                    if not np.allclose(self.x, prev_param["last_requested_point"]):
                        raise ValueError(
                            "Values for input variables do not match requested points: {} != {}".format(
                                str(self.x), str(prev_param["last_requested_point"])
                            )
                        )
                    x, state = self.step(sess, self.x, self.y, state)
                    self.saver.save(sess, self._infer_model_path)
            return x, state
