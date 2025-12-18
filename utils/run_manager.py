import os
import csv
import json
from datetime import datetime


class RunManager:
    def __init__(self, env: str, config: dict):
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_id = f"{ts}_{env}"
        self.run_dir = os.path.join("runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.plot_path = os.path.join(self.run_dir, "plot.png")

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "score", "mean_score", "epsilon", "loss"])

    def log(self, episode: int, score: int, mean_score: float, epsilon: float, loss):
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([episode, score, f"{mean_score:.4f}", f"{epsilon:.4f}", "" if loss is None else f"{loss:.6f}"])

    @staticmethod
    def models_dir():
        os.makedirs("models", exist_ok=True)
        return "models"

    @staticmethod
    def model_path(filename: str):
        return os.path.join(RunManager.models_dir(), filename)
