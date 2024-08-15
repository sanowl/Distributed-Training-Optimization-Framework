import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from typing import Callable, Dict, Any, List, Union, Optional
from distributed_utils import DistributedUtils
import joblib
import os
import logging
from functools import partial

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
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.study_name = study_name
        self.direction = direction
        self.sampler = sampler or TPESampler()
        self.pruner = pruner or MedianPruner()
        self.storage = storage or f"sqlite:///{study_name}.db"
        
        DistributedUtils.initialize_distributed()
        
        if DistributedUtils.is_main_process():
            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage,
                load_if_exists=True
            )
        DistributedUtils.sync_all_processes()

    def optimize(self, n_jobs: int = -1, timeout: Optional[int] = None) -> Dict[str, Any]:
        if DistributedUtils.is_main_process():
            logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials.")
            self.study.optimize(
                self.objective_fn, 
                n_trials=self.n_trials, 
                n_jobs=n_jobs,
                timeout=timeout
            )
            best_params = self.study.best_params
        else:
            best_params = None

        DistributedUtils.sync_all_processes()
        best_params = DistributedUtils.broadcast(best_params)
        return best_params

    def get_best_trial(self) -> optuna.Trial:
        return self.study.best_trial

    def log_best_trial(self) -> None:
        if DistributedUtils.is_main_process():
            best_trial = self.get_best_trial()
            logger.info(f"Best trial:")
            logger.info(f"  Value: {best_trial.value}")
            logger.info(f"  Params: {best_trial.params}")

    def save_study(self, file_path: str) -> None:
        if DistributedUtils.is_main_process():
            joblib.dump(self.study, file_path)
            logger.info(f"Study saved to {file_path}")

    def load_study(self, file_path: str) -> None:
        if DistributedUtils.is_main_process():
            self.study = joblib.load(file_path)
            logger.info(f"Study loaded from {file_path}")
        DistributedUtils.sync_all_processes()

    def plot_optimization_history(self) -> None:
        if DistributedUtils.is_main_process():
            import optuna.visualization as vis
            fig = vis.plot_optimization_history(self.study)
            fig.show()

    def plot_param_importances(self) -> None:
        if DistributedUtils.is_main_process():
            import optuna.visualization as vis
            fig = vis.plot_param_importances(self.study)
            fig.show()

    def plot_parallel_coordinate(self) -> None:
        if DistributedUtils.is_main_process():
            import optuna.visualization as vis
            fig = vis.plot_parallel_coordinate(self.study)
            fig.show()

    def plot_slice(self) -> None:
        if DistributedUtils.is_main_process():
            import optuna.visualization as vis
            fig = vis.plot_slice(self.study)
            fig.show()

    def get_trials_dataframe(self) -> Optional[pd.DataFrame]:
        if DistributedUtils.is_main_process():
            return self.study.trials_dataframe()
        return None

    def checkpoint(self, file_path: str) -> None:
        if DistributedUtils.is_main_process():
            joblib.dump(self.study, file_path)
            logger.info(f"Checkpoint saved to {file_path}")

    def resume_from_checkpoint(self, file_path: str) -> None:
        if DistributedUtils.is_main_process():
            self.study = joblib.load(file_path)
            logger.info(f"Resumed from checkpoint {file_path}")
        DistributedUtils.sync_all_processes()

    @staticmethod
    def objective_wrapper(func: Callable, trial: optuna.Trial) -> Union[float, List[float]]:
        try:
            return func(trial)
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            raise optuna.exceptions.TrialPruned()

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

    if DistributedUtils.is_main_process():
        search.plot_optimization_history()
        search.plot_param_importances()
        search.plot_parallel_coordinate()
        search.plot_slice()

        search.save_study("final_study.pkl")

        df = search.get_trials_dataframe()
        if df is not None:
            print(df.head())

    DistributedUtils.finalize()
