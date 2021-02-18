from summit.benchmarks.experimental_emulator import *
from summit.utils.dataset import DataSet

import pandas as pd
import matplotlib.pyplot as plt
import logging
import pkg_resources
import pathlib
from tqdm import trange
import argparse

DATA_PATH = pathlib.Path(pkg_resources.resource_filename("summit", "benchmarks/data"))
MODELS_PATH = pathlib.Path(
    pkg_resources.resource_filename("summit", "benchmarks/models")
)
SUMMARY_FILE = "README.md"
MAX_EPOCHS = 1000
CV_FOLDS = 5


def train_reizman(show_plots=False):
    results = [
        train_one_reizman(i, show_plots=show_plots)
        for i in trange(1, 5, desc="Reizman")
    ]

    # Average scores from cross validation
    results_average = [
        {f"avg_{score_name}": scores.mean() for score_name, scores in result.items()}
        for result in results
    ]
    index = [f"case_{i}" for i in range(1, 5)]

    results_df = pd.DataFrame.from_records(results_average, index=index)
    results_df.index.rename("case", inplace=True)
    results_df.to_csv(f"results/reizman_suzuki_scores.csv")


def train_one_reizman(case, show_plots=False, save_plots=True):
    # Setup
    model_name = f"reizman_suzuki_case_{case}"
    domain = ReizmanSuzukiEmulator.setup_domain()
    ds = DataSet.read_csv(DATA_PATH / f"{model_name}.csv")

    # Create emulator and train
    exp = ExperimentalEmulator(
        model_name,
        domain,
        dataset=ds,
        regressor=ANNRegressor,
    )
    res = exp.train(
        max_epochs=MAX_EPOCHS, cv_folds=CV_FOLDS, random_state=100, test_size=0.2
    )

    # Run test
    res_test = exp.test()
    res.update(res_test)

    # Save emulator
    model_path = pathlib.Path(MODELS_PATH / model_name)
    model_path.mkdir(exist_ok=True)
    exp.save(model_path)

    # Make plot for posteriority sake
    fig, ax = exp.parity_plot(include_test=True)
    if save_plots:
        fig.savefig(f"results/{model_name}.png", dpi=100)
    if show_plots:
        plt.show()

    return res


def train_baumgartner(show_plots=False):
    # Train model using one-hot encoding for categorical
    print("Training Baumgartner model")
    result = train_baumgartner_no_descriptors()
    results_average = [
        {f"avg_{score_name}": scores.mean() for score_name, scores in result.items()}
    ]

    index = ["one-hot"]
    results_df = pd.DataFrame.from_records(results_average, index=index)
    results_df.index.rename("case", inplace=True)
    results_df.to_csv(f"results/baumgartner_aniline_cn_crosscoupling_scores.csv")


def train_baumgartner_no_descriptors(show_plots=False, save_plots=True):
    # Setup
    model_name = f"baumgartner_aniline_cn_crosscoupling"
    domain = BaumgartnerCrossCouplingEmulator.setup_domain()
    ds = DataSet.read_csv(DATA_PATH / f"{model_name}.csv")

    # Create emulator and train
    exp = ExperimentalEmulator(
        model_name,
        domain,
        dataset=ds,
        regressor=ANNRegressor,
        output_variable_names=["yield"],
    )
    res = exp.train(
        max_epochs=MAX_EPOCHS, cv_folds=CV_FOLDS, random_state=100, test_size=0.2
    )

    # # Run test
    res_test = exp.test()
    res.update(res_test)

    # Save emulator
    model_path = pathlib.Path(MODELS_PATH / model_name)
    model_path.mkdir(exist_ok=True)
    exp.save(model_path)

    # Make plot for posteriority sake
    fig, ax = exp.parity_plot(include_test=True)
    if save_plots:
        fig.savefig(f"results/{model_name}.png", dpi=100)
    if show_plots:
        plt.show()

    return res


def create_markdown():
    # Create markdown report
    md = (
        "# Train Emulators\n"
        "<!-- This file is auto-generated. Do not edit directly."
        "You can regenerate using python train_emulators.py --bypass-training -->\n"
        "The `train_emulators.py` script will train emulators and create this report.\n"
    )

    # Reizman
    reizman_text = (
        "## Reizman Suzuki Cross coupling \n"
        "This is the data from training of the reizman suzuki benchmark "
        f"for {MAX_EPOCHS} epochs with {CV_FOLDS} cross-validation folds.\n"
    )
    baumgartner_text = (
        "## Baumgartner C-N Cross Cross Coupling \n"
        "This is the data from training of the Baumgartner C-N aniline cross-coupling benchmark "
        f"for {MAX_EPOCHS} epochs with {CV_FOLDS} cross-validation folds.\n"
    )
    texts = [reizman_text, baumgartner_text]
    df_reizman = pd.read_csv("results/reizman_suzuki_scores.csv")
    df_baumgartner = pd.read_csv(
        "results/baumgartner_aniline_cn_crosscoupling_scores.csv"
    )
    dfs = [df_reizman, df_baumgartner]

    for text, df in zip(texts, dfs):
        rename = dict()
        for column in df.columns:
            mse_substring = "neg_root_mean_squared_error"
            if mse_substring in column:
                rename[column] = column.replace(mse_substring, "RMSE")
                df[column] = -1.0 * df[column]
        df = df.rename(columns=rename)
        df = df.drop(columns="avg_score_time")
        md += text
        md += df.round(2).to_markdown(index=False)
        md += "\n"

    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--bypass_training", action="store_true")
    args = parser.parse_args()

    # Training
    if not args.bypass_training:
        train_reizman()
        train_baumgartner()

    # Create report
    md = create_markdown()
    with open("README.md", "w") as f:
        f.write(md)
