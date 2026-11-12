"""Track the runs of every notebook with MLflow.

    from src.models import tracking
    tracking.setup("baselines")                       # once per notebook: the experiment
    with tracking.start_run("logreg_binary", params):
        ...                                           # train the model
        tracking.log_metrics(binary_metrics(y_val, p))
        tracking.log_model(model, input_example=x_val[:5])

fit() in train.py already does this for the neural models: one run per training, with
the metrics of every epoch.

The runs are stored in mlruns/ at the repository root (not committed): mlruns/mlflow.db
holds the runs with their parameters and metrics, mlruns/artifacts/ the logged files
and models. To browse them, run from the repository root

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

and open http://127.0.0.1:5000.
"""
import contextlib
import copy
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

ROOT = Path(__file__).resolve().parents[2]
MLRUNS = ROOT / "mlruns"

_experiment = None   # set by setup()


def setup(experiment, mlruns=MLRUNS):
    """Log the next runs to `experiment`, stored in the folder `mlruns`."""
    global _experiment
    mlruns = Path(mlruns)
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(mlruns / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(mlruns / "artifacts" / experiment).as_uri())
    mlflow.set_experiment(experiment)
    _experiment = experiment


def start_run(name, params=None, folder=None):
    """Start the run `name` and log its `params`; use it as `with start_run(...):`.

    With `folder` (the checkpoint folder of a training), the run id is kept in
    folder/mlflow_run_id.txt, so a training that continues after an interruption
    continues the same run (its parameters were logged when it started).
    """
    if _experiment is None:
        raise RuntimeError("call tracking.setup(experiment) first")
    saved = Path(folder) / "mlflow_run_id.txt" if folder is not None else None
    if saved is not None and saved.exists():
        try:
            return mlflow.start_run(run_id=saved.read_text().strip())
        except MlflowException:
            pass   # the run is gone (e.g. mlruns/ was deleted): start a new one
    run = mlflow.start_run(run_name=name)
    if saved is not None:
        saved.write_text(run.info.run_id)
    if params:
        mlflow.log_params(params)
    return run


def log_metrics(metrics, step=None):
    """Log numbers to the active run; `step` (e.g. the epoch) makes a curve per metric."""
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)


def log_files(*paths):
    """Attach files to the active run, e.g. a history.csv."""
    for path in paths:
        mlflow.log_artifact(str(path))


def log_figure(fig, name):
    """Attach a matplotlib figure to the active run, as `name` (e.g. losses.png)."""
    mlflow.log_figure(fig, name)


def log_model(model, input_example, name="model"):
    """Log a trained model to the active run; the run page then lists it under Models.

    A PyTorch model is saved as a traced graph (pt2, MLflow's safe format), which
    loads without this code base: mlflow.pytorch.load_model(uri)(x) gives the logits
    of every output, as model(x) does. A scikit-learn model is saved with
    mlflow.sklearn, and loads with mlflow.sklearn.load_model(uri).
    `input_example`: a few inputs of the model (encoded domains for the CNN).
    """
    torch = sys.modules.get("torch")   # a model can only be a PyTorch model if torch is loaded
    if torch is not None and isinstance(model, torch.nn.Module):
        model = copy.deepcopy(model).cpu().eval()   # traced on the CPU, wherever it trained
        example = np.asarray(input_example)
        signature = ModelSignature(
            inputs=Schema([TensorSpec(example.dtype, (-1, *example.shape[1:]))]))
        # MLflow also checks the example for model serving, which expects a single output
        # tensor; the CNN returns a dict of logits per output, so that check only warns.
        with _quiet("mlflow.models.model"):
            return mlflow.pytorch.log_model(model, name=name, serialization_format="pt2",
                                            input_example=example, signature=signature)
    return mlflow.sklearn.log_model(model, name=name, input_example=input_example)


@contextlib.contextmanager
def _quiet(logger):
    """Hide the warnings of one logger for the duration of the block."""
    log = logging.getLogger(logger)
    level = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(level)
