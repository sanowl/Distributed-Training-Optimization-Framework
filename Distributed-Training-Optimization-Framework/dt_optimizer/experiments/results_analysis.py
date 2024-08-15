import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import shutil
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultsManager:
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.version_dir = os.path.join(results_dir, "versions")
        os.makedirs(self.version_dir, exist_ok=True)
        
    def save_results(self, experiment_name: str, results: Dict[str, Any], version: Optional[str] = None) -> None:
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_path = os.path.join(self.version_dir, version)
        os.makedirs(version_path, exist_ok=True)
        
        results_path = os.path.join(version_path, f"{experiment_name}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {results_path}")

        # Update latest version
        latest_path = os.path.join(self.results_dir, f"{experiment_name}.json")
        shutil.copy2(results_path, latest_path)

    def load_results(self, experiment_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if version:
            results_path = os.path.join(self.version_dir, version, f"{experiment_name}.json")
        else:
            results_path = os.path.join(self.results_dir, f"{experiment_name}.json")
        
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            logger.info(f"Results loaded from {results_path}")
            return results
        else:
            logger.warning(f"No results found for experiment: {experiment_name}")
            return None

    def list_results(self, filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> pd.DataFrame:
        all_results = []
        for file_name in os.listdir(self.results_dir):
            if file_name.endswith(".json"):
                experiment_name = file_name.replace(".json", "")
                result = self.load_results(experiment_name)
                if result:
                    if filter_func is None or filter_func(result):
                        all_results.append({
                            "Experiment": experiment_name,
                            "Metrics": result.get("metrics", {}),
                            "Hyperparameters": result.get("hyperparameters", {})
                        })
        df = pd.DataFrame(all_results)
        return df

    def compare_results(self, metric_names: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        
        results_list = []
        for file_name in os.listdir(self.results_dir):
            if file_name.endswith(".json"):
                experiment_name = file_name.replace(".json", "")
                result = self.load_results(experiment_name)
                if result:
                    experiment_data = {"Experiment": experiment_name}
                    for metric_name in metric_names:
                        if metric_name in result.get("metrics", {}):
                            experiment_data[metric_name] = result["metrics"][metric_name]
                    results_list.append(experiment_data)
        
        df = pd.DataFrame(results_list)
        return df.sort_values(by=metric_names[0], ascending=True)

    def plot_results(self, metric_names: Union[str, List[str]], plot_type: str = "bar") -> None:
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        
        df = self.compare_results(metric_names)
        
        if plot_type == "bar":
            df.set_index("Experiment").plot(kind="bar", figsize=(12, 6))
        elif plot_type == "line":
            df.set_index("Experiment").plot(kind="line", marker="o", figsize=(12, 6))
        elif plot_type == "scatter":
            if len(metric_names) != 2:
                raise ValueError("Scatter plot requires exactly two metrics")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=metric_names[0], y=metric_names[1])
            for i, txt in enumerate(df["Experiment"]):
                plt.annotate(txt, (df[metric_names[0]].iloc[i], df[metric_names[1]].iloc[i]))
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        plt.title(f"Comparison of {', '.join(metric_names)} across experiments")
        plt.xlabel("Experiment" if plot_type != "scatter" else metric_names[0])
        plt.ylabel(metric_names[0] if len(metric_names) == 1 else "Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def get_best_experiment(self, metric_name: str, higher_is_better: bool = True) -> Optional[str]:
        df = self.compare_results(metric_name)
        if df.empty:
            return None
        if higher_is_better:
            return df.loc[df[metric_name].idxmax(), "Experiment"]
        else:
            return df.loc[df[metric_name].idxmin(), "Experiment"]

    def get_experiment_history(self, experiment_name: str) -> pd.DataFrame:
        versions = []
        for version in os.listdir(self.version_dir):
            result = self.load_results(experiment_name, version)
            if result:
                result["Version"] = version
                versions.append(result)
        return pd.DataFrame(versions).sort_values("Version")

    def export_results(self, output_file: str, format: str = "csv") -> None:
        df = self.list_results()
        if format == "csv":
            df.to_csv(output_file, index=False)
        elif format == "excel":
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        logger.info(f"Results exported to {output_file}")

    def delete_experiment(self, experiment_name: str) -> None:
        experiment_path = os.path.join(self.results_dir, f"{experiment_name}.json")
        if os.path.exists(experiment_path):
            os.remove(experiment_path)
            logger.info(f"Deleted experiment: {experiment_name}")
        else:
            logger.warning(f"Experiment not found: {experiment_name}")

        for version in os.listdir(self.version_dir):
            version_path = os.path.join(self.version_dir, version, f"{experiment_name}.json")
            if os.path.exists(version_path):
                os.remove(version_path)
                logger.info(f"Deleted version {version} of experiment: {experiment_name}")

    def get_experiment_summary(self) -> pd.DataFrame:
        summary = defaultdict(list)
        for file_name in os.listdir(self.results_dir):
            if file_name.endswith(".json"):
                experiment_name = file_name.replace(".json", "")
                result = self.load_results(experiment_name)
                if result:
                    summary["Experiment"].append(experiment_name)
                    for key, value in result.get("metrics", {}).items():
                        summary[f"Metric: {key}"].append(value)
                    for key, value in result.get("hyperparameters", {}).items():
                        summary[f"Param: {key}"].append(value)
        return pd.DataFrame(summary)

# Example usage
if __name__ == "__main__":
    results_manager = ResultsManager("./experiment_results")

    # Save some example results
    results_manager.save_results("experiment1", {
        "metrics": {"accuracy": 0.85, "f1_score": 0.82},
        "hyperparameters": {"learning_rate": 0.001, "batch_size": 32}
    })
    results_manager.save_results("experiment2", {
        "metrics": {"accuracy": 0.88, "f1_score": 0.86},
        "hyperparameters": {"learning_rate": 0.0005, "batch_size": 64}
    })

    # List all results
    print(results_manager.list_results())

    # Compare specific metrics
    print(results_manager.compare_results(["accuracy", "f1_score"]))

    # Plot results
    results_manager.plot_results(["accuracy", "f1_score"], plot_type="scatter")

    # Get best experiment
    best_exp = results_manager.get_best_experiment("accuracy")
    print(f"Best experiment based on accuracy: {best_exp}")

    # Get experiment history
    print(results_manager.get_experiment_history("experiment1"))

    # Export results
    results_manager.export_results("all_results.csv")

    # Get experiment summary
    print(results_manager.get_experiment_summary())

    # Delete an experiment
    results_manager.delete_experiment("experiment1")

    print("Results Manager demonstration completed!")