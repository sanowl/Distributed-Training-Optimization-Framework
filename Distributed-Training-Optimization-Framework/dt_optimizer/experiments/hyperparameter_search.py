import os
import joblib
import logging
import optuna
import pandas as pd
from typing import Callable, Dict, Any, List, Union, Optional
from functools import partial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from distributed_utils import DistributedUtils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistributedHyperparameterSearch:
    def __init__(self, 
                 objective_fn: Union[Callable[[optuna.Trial], float], Callable[[optuna.Trial], List[float]]],
                 n_trials: int = 50,
                 study_name: str = "distributed_study",
                 direction: Union[str, List[str]] = "minimize",
                 sampler: Optional[optuna.samplers.BaseSampler] = None,
                 pruner: Optional[optuna.pruners.BasePruner] = None,
                 storage: Optional[str] = None):
        """
        Initialize the Distributed Hyperparameter Search.

        Args:
            objective_fn (Callable): The objective function to optimize.
            n_trials (int): The number of trials for optimization.
            study_name (str): The name of the study.
            direction (Union[str, List[str]]): Direction to optimize ('minimize' or 'maximize').
            sampler (Optional[optuna.samplers.BaseSampler]): The sampler to use for hyperparameter sampling.
            pruner (Optional[optuna.pruners.BasePruner]): The pruner to use for early stopping.
            storage (Optional[str]): The storage backend for saving the study.
        """
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.study_name = study_name
        self.direction = direction
        self.sampler = sampler or TPESampler()
        self.pruner = pruner or MedianPruner()
        self.storage = storage or f"sqlite:///{study_name}.db"
        
        DistributedUtils.initialize_distributed()
        
        main_process = DistributedUtils.is_main_process()
        self.study = main_process and optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True
        )
        DistributedUtils.sync_all_processes()

    def optimize(self, n_jobs: int = -1, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the optimization process.

        Args:
            n_jobs (int): Number of parallel jobs.
            timeout (Optional[int]): Time limit for the optimization.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        main_process = DistributedUtils.is_main_process()
        best_params = main_process and self._run_optimization(n_jobs, timeout)
        DistributedUtils.sync_all_processes()
        return DistributedUtils.broadcast(best_params)

    def _run_optimization(self, n_jobs, timeout):
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials.")
        self.study.optimize(
            self.objective_fn, 
            n_trials=self.n_trials, 
            n_jobs=n_jobs,
            timeout=timeout
        )
        return self.study.best_params

    def get_best_trial(self) -> optuna.Trial:
        """Retrieve the best trial found during optimization."""
        return self.study.best_trial

    def log_best_trial(self) -> None:
        """Log the details of the best trial."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._log_best_trial()

    def _log_best_trial(self):
        best_trial = self.get_best_trial()
        logger.info("Best trial:")
        logger.info(f"  Value: {best_trial.value}")
        logger.info(f"  Params: {best_trial.params}")

    def save_study(self, file_path: str) -> None:
        """Save the current state of the study to a file."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._save_study(file_path)

    def _save_study(self, file_path):
        joblib.dump(self.study, file_path)
        logger.info(f"Study saved to {file_path}")

    def load_study(self, file_path: str) -> None:
        """Load a study from a file."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._load_study(file_path)
        DistributedUtils.sync_all_processes()

    def _load_study(self, file_path):
        self.study = joblib.load(file_path)
        logger.info(f"Study loaded from {file_path}")

    def plot_optimization_history(self) -> None:
        """Plot the optimization history of the study."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._plot_optimization_history()

    def _plot_optimization_history(self):
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(self.study)
        fig.show()

    def plot_param_importances(self) -> None:
        """Plot the importance of hyperparameters."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._plot_param_importances()

    def _plot_param_importances(self):
        import optuna.visualization as vis
        fig = vis.plot_param_importances(self.study)
        fig.show()

    def plot_parallel_coordinate(self) -> None:
        """Plot parallel coordinates for hyperparameters."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._plot_parallel_coordinate()

    def _plot_parallel_coordinate(self):
        import optuna.visualization as vis
        fig = vis.plot_parallel_coordinate(self.study)
        fig.show()

    def plot_slice(self) -> None:
        """Plot a slice of hyperparameter search space."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._plot_slice()

    def _plot_slice(self):
        import optuna.visualization as vis
        fig = vis.plot_slice(self.study)
        fig.show()

    def get_trials_dataframe(self) -> Optional[pd.DataFrame]:
        """Get a DataFrame of the trials."""
        main_process = DistributedUtils.is_main_process()
        return main_process and self.study.trials_dataframe()

    def checkpoint(self, file_path: str) -> None:
        """Save a checkpoint of the study."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._checkpoint(file_path)

    def _checkpoint(self, file_path):
        joblib.dump(self.study, file_path)
        logger.info(f"Checkpoint saved to {file_path}")

    def resume_from_checkpoint(self, file_path: str) -> None:
        """Resume the study from a checkpoint."""
        main_process = DistributedUtils.is_main_process()
        main_process and self._resume_from_checkpoint(file_path)
        DistributedUtils.sync_all_processes()

    def _resume_from_checkpoint(self, file_path):
        self.study = joblib.load(file_path)
        logger.info(f"Resumed from checkpoint {file_path}")

    @staticmethod
    def objective_wrapper(func: Callable, trial: optuna.Trial) -> Union[float, List[float]]:
        """
        Wrapper for the objective function to handle exceptions.

        Args:
            func (Callable): The objective function.
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            Union[float, List[float]]: The result of the objective function.
        """
        try:
            return func(trial)
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            raise optuna.exceptions.TrialPruned()

# Example usage of DistributedHyperparameterSearch
def example_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return x**2 + y**2

if __name__ == "__main__":
    search = DistributedHyperparameterSearch(
        objective_fn=partial(DistributedHyperparameterSearch.objective_wrapper, example_objective),
        n_trials=100,
        study_name="example_distributed_study",
        direction="minimize",
        sampler=TPESampler(n_startup_trials=10),
        pruner=HyperbandPruner()
    )

    best_params = search.optimize(n_jobs=-1, timeout=600)
    search.log_best_trial()

    main_process = DistributedUtils.is_main_process()
    main_process and search._plot_optimization_history()
    main_process and search._plot_param_importances()
    main_process and search._plot_parallel_coordinate()
    main_process and search._plot_slice()

    main_process and search._save_study("final_study.pkl")

    df = main_process and search.get_trials_dataframe()
    df and print(df.head())

    DistributedUtils.finalize()
