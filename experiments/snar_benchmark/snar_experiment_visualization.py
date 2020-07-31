from summit import Runner
from summit.utils.multiobjective import pareto_efficient, hypervolume

from neptune.sessions import Session, HostedNeptuneBackend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import os
import zipfile
import shutil
import warnings
from textwrap import wrap
import collections


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class PlotExperiments:
    def __init__(
        self, project: str, experiment_ids: list, tag: list = None, state: list = None
    ):
        self.session = Session(backend=HostedNeptuneBackend())
        self.proj = self.session.get_project(project)
        self.runners = {}
        self.experiment_ids = experiment_ids
        self.tag = tag
        self.state = state
        self._restore_runners()
        self._create_param_df()

    def _restore_runners(self):
        """Restore runners from Neptune Artifacts"""
        # Download artifacts
        n_experiments = len(self.experiment_ids)
        experiments = []
        if n_experiments > 100:
            for i in range(n_experiments // 100):
                experiments += self.proj.get_experiments(
                    id=self.experiment_ids[i * 100 : (i + 1) * 100],
                    tag=self.tag,
                    state=self.state,
                )
            remainder = n_experiments % 100
            experiments += self.proj.get_experiments(
                id=self.experiment_ids[(i + 1) * 100 : (i + 1) * 100 + remainder],
                tag=self.tag,
                state=self.state,
            )
        else:
            experiments = self.proj.get_experiments(
                id=self.experiment_ids, tag=self.tag, state=self.state
            )
        for experiment in experiments:
            path = f"data/{experiment.id}"
            try:
                os.mkdir(path,)
            except FileExistsError:
                pass
            experiment.download_artifacts(destination_dir=path)

            # Unzip somehow
            files = os.listdir(path)
            with zipfile.ZipFile(path + "/" + files[0], "r") as zip_ref:
                zip_ref.extractall(path)

            # Get filename
            path += "/output"
            files = os.listdir(path)
            files_json = [f for f in files if ".json" in f]

            # Restore runner
            r = Runner.load(path + "/" + files_json[0])
            self.runners[experiment.id] = r

            # Remove file
            shutil.rmtree(f"data/{experiment.id}")

    def _create_param_df(self, reference=[-2957, 10.7]):
        """Create a parameters dictionary
        
        Parameters
        ----------
        reference : array-like, optional
            Reference for the hypervolume calculatio

        """
        records = []
        for experiment_id, r in self.runners.items():
            record = {}
            record["experiment_id"] = experiment_id

            # Transform
            transform_name = r.strategy.transform.__class__.__name__
            transform_params = r.strategy.transform.to_dict()["transform_params"]
            record["transform_name"] = transform_name
            if transform_name == "Chimera":
                hierarchy = transform_params["hierarchy"]
                for objective_name, v in hierarchy.items():
                    key = f"{objective_name}_tolerance"
                    record[key] = v["tolerance"]
            elif transform_name == "MultitoSingleObjective":
                record.update(transform_params)

            # Strategy
            record["strategy_name"] = r.strategy.__class__.__name__

            # Batch size
            record["batch_size"] = r.batch_size

            # Number of initial experiments
            try:
                record["num_initial_experiments"] = r.n_init
            except AttributeError:
                pass

            # Terminal hypervolume
            data = r.experiment.data[["sty", "e_factor"]].to_numpy()
            data[:, 0] *= -1  # make it a minimzation problem
            y_front, _ = pareto_efficient(data, maximize=False)
            hv = hypervolume(y_front, ref=reference)
            record["terminal_hypervolume"] = hv

            # Computation time
            time = r.experiment.data["computation_t"].sum()
            record["computation_t"] = time

            records.append(record)

        # Make pandas dataframe
        self.df = pd.DataFrame.from_records(records)
        return self.df

    def best_pareto_grid(self, ncols=3, figsize=(20, 40)):
        """Make a grid of pareto plots

        Only includes the run with the maximum terminal hypervolume for each 
        unique combination.

        Parameters
        ----------
        ncols : int, optional
            The number of columns in the grid. Defaults to 3
        figsize : tuple, optional
            The figure size. Defaults to 20 wide x 40 high

        """
        # Group experiment repeats
        df = self.df.copy()
        df = df.set_index("experiment_id")
        df = df.drop(columns=["terminal_hypervolume"])
        uniques = df.drop_duplicates(keep="last")  # This actually groups them
        uniques = uniques.sort_values(by=["strategy_name", "transform_name"])
        df_new = self.df.copy()

        nrows = len(uniques) // ncols
        nrows += 1 if len(uniques) % ncols != 0 else 0

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(wspace=0.2, hspace=0.5)
        i = 1
        # Loop through groups of repeatss
        for index, unique in uniques.iterrows():
            # Find number of matching rows to this unique row
            temp_df = df_new.merge(unique.to_frame().transpose(), how="inner")

            # Find experiment with maximum hypervolume
            max_hv_index = temp_df["terminal_hypervolume"].argmax()
            experiment_id = temp_df.iloc[max_hv_index]["experiment_id"]

            # Get runner
            r = self.runners[experiment_id]

            # Create pareto plot
            ax = plt.subplot(nrows, ncols, i)
            r.experiment.pareto_plot(ax=ax)
            title = self._create_label(unique)
            title = "\n".join(wrap(title, 30))
            ax.set_title(title)
            ax.set_xlabel(r"Space Time Yield ($kg \; m^{-3} h^{-1}$)")
            ax.set_ylabel("E-factor")
            ax.set_xlim(0, 1.2e4)
            ax.set_ylim(0, 70)
            i += 1

        return fig

    def plot_hv_trajectories(
        self,
        trajectory_length,
        reference=[-2957, 10.7],
        plot_type="matplotlib",
        include_experiment_ids=False,
        min_terminal_hv_avg=0,
    ):
        """ Plot the hypervolume trajectories with repeats as 95% confidence interval
        
        Parameters
        ----------
        reference : array-like, optional
            Reference for the hypervolume calculation. Defaults to -2957, 10.7
        plot_type : str, optional
            Plotting backend to use: matplotlib or plotly. Defaults to matplotlib.
        include_experiment_ids : bool, optional
            Whether to include experiment ids in the plot labels
        min_terminal_hv_avg : float, optional`
            Minimum terminal average hypervolume cutoff for inclusion in the plot. Defaults to 0.
        """
        # Create figure
        if plot_type == "matplotlib":
            fig, ax = plt.subplots(1)
        elif plot_type == "plotly":
            fig = go.Figure()
        else:
            raise ValueError(
                f"{plot_type} is not a valid plot type. Must be matplotlib or plotly."
            )

        # Group experiment repeats
        df = self.df.copy()
        df = df.set_index("experiment_id")
        df = df.drop(columns=["terminal_hypervolume"])
        uniques = df.drop_duplicates(keep="last")  # This actually groups them
        df_new = self.df.copy()

        colors = px.colors.qualitative.Plotly
        cycle = len(colors)
        c_num = 0
        self.hv = {}
        for index, unique in uniques.iterrows():
            # Find number of matching rows to this unique row
            temp_df = df_new.merge(unique.to_frame().transpose(), how="inner")
            ids = temp_df["experiment_id"].values

            # Calculate hypervolume trajectories
            hv_trajectories = np.zeros([trajectory_length, len(ids)])
            for j, experiment_id in enumerate(ids):
                r = self.runners[experiment_id]
                data = r.experiment.data[["sty", "e_factor"]].to_numpy()
                data[:, 0] *= -1  # make it a minimzation problem
                for i in range(trajectory_length):
                    y_front, _ = pareto_efficient(data[0 : i + 1, :], maximize=False)
                    hv_trajectories[i, j] = hypervolume(y_front, ref=reference)

            # Mean and standard deviation
            hv_mean_trajectory = np.mean(hv_trajectories, axis=1)
            hv_std_trajectory = np.std(hv_trajectories, axis=1)

            if hv_mean_trajectory[-1] < min_terminal_hv_avg:
                continue

            # Update plot
            t = np.arange(1, trajectory_length + 1)
            label = self._create_label(unique)
            if include_experiment_ids:
                label += f" ({ids[0]}-{ids[-1]})"

            lower = hv_mean_trajectory - 1.96 * hv_std_trajectory
            lower = np.clip(lower, 0, None)
            upper = hv_mean_trajectory + 1.96 * hv_std_trajectory
            if plot_type == "matplotlib":
                ax.plot(t, hv_mean_trajectory, label=label)
                ax.fill_between(t, lower, upper, alpha=0.1)
            elif plot_type == "plotly":
                r, g, b = hex_to_rgb(colors[c_num])
                color = lambda alpha: f"rgba({r},{g},{b},{alpha})"
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=hv_mean_trajectory,
                        mode="lines",
                        name=label,
                        line=dict(color=color(1)),
                        legendgroup=label,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=lower,
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0),
                        fillcolor=color(0.1),
                        showlegend=False,
                        legendgroup=label,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=upper,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(width=0),
                        fillcolor=color(0.1),
                        showlegend=False,
                        legendgroup=label,
                    )
                )
            if cycle == c_num + 1:
                c_num = 0
            else:
                c_num += 1

        # Plot formattting
        if plot_type == "matplotlib":
            ax.set_xlabel("Experiments")
            ax.set_ylabel("Hypervolume")
            legend = ax.legend(loc=(1.2, 0.5))
            ax.tick_params(direction="in")
            ax.set_xlim(1, trajectory_length)
            return fig, ax, legend
        elif plot_type == "plotly":
            fig.update_layout(
                xaxis=dict(title="Experiments"), yaxis=dict(title="Hypervolume")
            )
            fig.show()
            return fig

    def _create_label(self, unique):
        transform_text = (
            unique["transform_name"]
            if unique["transform_name"] != "Transform"
            else "No transform"
        )
        chimera_params = f" (STY tol.={unique['sty_tolerance']}, E-factor tol.={unique['e_factor_tolerance']})"
        transform_text += (
            chimera_params if unique["transform_name"] == "Chimera" else ""
        )
        final_text = f"{unique['strategy_name']}, {transform_text}, {unique['num_initial_experiments']} initial experiments"
        if unique["num_initial_experiments"] == 1:
            final_text = final_text.rstrip("s")
        return final_text

    # def time_hv_bar_plot(self):
    #     # Create figures
    #     fig, ax = plt.subplots(1)

    #     # Group experiment repeats
    #     df = self.df.copy()
    #     df = df.set_index("experiment_id")
    #     df = df.drop(columns=['terminal_hypervolume'])
    #     uniques = df.drop_duplicates(keep="last")  # This actually groups them
    #     uniques = uniques.sort_values(["strategy_name", "transform_name"])
    #     df_new = self.df.copy()
    #     df_new = df_new.set_index("experiment_id")

    #     # Get all strategies
    #     strategies = self.df["strategy_name"].drop_duplicates()
    #     strategies = strategies.sort_values(ascending=True)

    #     avg_times = []
    #     std_times = []
    #     labels = []
    #     for strategy in strategies:
    #         # Find all unique combinations for this strategy
    #         strategy_matches = uniques[uniques['strategy_name'] == strategy]

    #         # Find combination with maximum average terminal hyperovlume
    #         for i, combo in strategy_matches.iterrows():
    #             df_new.
    #         ids = temp_df["experiment_id"].values

    #         hv_df = df.set_index('experiment_id')

    #         times = np.zeros(len(ids))

    #         for i, experiment_id in enumerate(ids):
    #             r = self.runners[experiment_id]
    #             times[i] = r.experiment.data['computation_t'].sum()

    #         times = np.array(times)/60 #convert to minutes
    #         avg_time = np.mean(times)
    #         std_time = np.std(times)

    #         avg_times.append(avg_time)
    #         std_times.append(std_time)
    #         labels.append(r.strategy.__class__.__name__)

    #     x = np.arange(0, len(avg_times))
    #     c = hex_to_rgb("#a50026")
    #     ax.bar(x, avg_times, yerr=std_times, tick_label=labels, color="#a50026")
    #     ax.set_ylabel("Average Optimisation Time (minutes)")
    #     ax.set_yscale('log')
    #     plt.xticks(rotation=45)
    #     return fig, ax

    def iterations_to_threshold(self, sty_threshold=1e4, e_factor_threshold=10.0):
        # Group experiment repeats
        df = self.df.copy()
        df = df.set_index("experiment_id")
        uniques = df.drop_duplicates(keep="last")  # This actually groups them
        df_new = self.df.copy()
        experiments = {}
        results = []
        uniques["mean_iterations"] = None
        uniques["std_iterations"] = None
        uniques["num_repeats"] = None
        # Find iterations to threshold
        for index, unique in uniques.iterrows():
            # Find number of matching rows to each unique row
            temp_df = df_new.merge(unique.to_frame().transpose(), how="inner")
            ids = temp_df["experiment_id"].values

            # Number of iterations calculation
            num_iterations = []
            something_happens = False
            for experiment_id in ids:
                data = self.runners[experiment_id].experiment.data[["sty", "e_factor"]]
                # Check if repeat matches threshold requirements
                meets_threshold = data[
                    (data["sty"] >= sty_threshold)
                    & (data["e_factor"] <= e_factor_threshold)
                ]
                # Calculate iterations to meet threshold
                if len(meets_threshold.index) > 0:
                    num_iterations.append(meets_threshold.index[0])
                    something_happens = True

            if something_happens:
                mean = np.mean(num_iterations)
                std = np.std(num_iterations)
                uniques["mean_iterations"][index] = mean
                uniques["std_iterations"][index] = std
                uniques["num_repeats"][index] = len(num_iterations)

        return uniques


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex to RGA
    From https://community.plotly.com/t/scatter-plot-fill-with-color-how-to-set-opacity-of-fill/29591
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
