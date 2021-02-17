from summit.benchmarks.experimental_emulator import (
    ExperimentalEmulator,
    ANNRegressor,
    ReizmanSuzukiEmulator,
    registry,
)
from summit.utils.dataset import DataSet
import matplotlib.pyplot as plt
import logging
import pkg_resources
import pathlib
from tqdm import trange

DATA_PATH = pathlib.Path(pkg_resources.resource_filename("summit", "benchmarks/data"))
MODELS_PATH = pathlib.Path(
    pkg_resources.resource_filename("summit", "benchmarks/models")
)


def test_train():
    logging.basicConfig(level=logging.INFO)
    model_name = f"reizman_suzuki_case_1"
    domain = ReizmanSuzukiEmulator.setup_domain()
    ds = DataSet.read_csv(DATA_PATH / f"{model_name}.csv")
    exp = ExperimentalEmulator(model_name, domain, dataset=ds, regressor=ANNRegressor)
    exp.train(max_epochs=1, cv_folds=5, random_state=100, test_size=0.2, verbose=1)
    # print(exp.predictors[0].regressor.named_steps.preprocessor.named_transformers_)
    # params = {
    #     "regressor__net__max_epochs": [200, 500, 1000]
    #     # "regressor__net__module__hidden_units": [32, 128, 512],
    #     # "regressor__net__module__num_hidden_layers": [1, 0],
    # }
    # res = exp.train(cv_folds=5, random_state=100, search_params=params, verbose=0)
    # new_params = exp.predictor.get_params()
    # for key in params.keys():
    #     logging.info(f"Selected number of hidden layers: {new_params[key]}")
    # fig, ax = exp.parity_plot(
    #     output_variables="yield", clip={"yield": (0, 100)}, include_test=True
    # )
    # plt.show()
    print(exp.to_dict())


def train_reizman():
    for i in trange(1, 5, desc="Reizman"):
        model_name = f"reizman_suzuki_case_{i}"
        domain = ReizmanSuzukiEmulator.setup_domain()
        ds = DataSet.read_csv(DATA_PATH / f"{model_name}.csv")
        exp = ExperimentalEmulator(
            model_name,
            domain,
            dataset=ds,
            regressor=ANNRegressor,
        )
        exp.train(max_epochs=1000, cv_folds=5, random_state=100)
        model_path = pathlib.Path(MODELS_PATH / model_name)
        model_path.mkdir(exist_ok=True)
        exp.save(model_path)
        fig, ax = exp.parity_plot(clip={"yield": (0, 100)}, include_test=True)
        fig.savefig(f"figures/{model_name}.png", dpi=100)
        plt.show()


def reproduce_bug():
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import Normalizer, StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_validate

    preprocessor = ColumnTransformer(
        [
            ("norm1", StandardScaler(), [0, 1, 2, 3, 4, 5, 6, 7]),
            ("norm2", Normalizer(norm="l2"), [8, 9, 10, 11, 12]),
        ]
    )
    model = LinearRegression()
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    # predictor = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    X, y = load_boston(return_X_y=True)
    # res = cross_validate(
    #     pipe,
    #     X,
    #     y,
    #     return_estimator=True,
    # )
    # print(
    #     res["estimator"][0]
    #     .regressor.named_steps.preprocessor._transformers[0][1]
    #     .transform(X)
    # )
    pipe.fit(X, y)
    # print(res["estimator"][0].named_steps.preprocessor._transformers[0][1].__dict__)
    print(pipe.named_steps.preprocessor.named_transformers_.norm1.mean_)
    # pipe.named_steps.preprocessor.named_transformers_


if __name__ == "__main__":
    # reproduce_bug()
    test_train()
