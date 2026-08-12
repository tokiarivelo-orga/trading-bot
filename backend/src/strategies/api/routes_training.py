"""Deep-learning model training: trigger, status, and honest history.

Process-wide and unprefixed by account, like the `backtest` and `indicators`
routers: there is one set of model weights on disk shared by every account,
so scoping these under `/accounts/{id}/` would imply an isolation that does
not exist.

The history endpoints deliberately expose the **out-of-sample** numbers, not
just "a model exists". A model with in-sample avg R 0.124 and out-of-sample
avg R -0.005 is not a model anyone should switch on, and the UI cannot show
that unless the API says it.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Request

from src.strategies.api.schemas import (
    StartTrainingRequest,
    TrainedModelOut,
    TrainingRunOut,
    WalkForwardFoldOut,
)
from src.strategies.application.model_training import (
    ModelTrainingService,
    TrainingAlreadyRunningError,
)

router = APIRouter(prefix="/model-training", tags=["model-training"])


def _service(request: Request) -> ModelTrainingService:
    service = getattr(request.app.state, "model_training", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="model training is not available in this process",
        )
    return service


@router.post(
    "/runs",
    response_model=TrainingRunOut,
    status_code=202,
    summary="Start a training run",
    description="Launches `scripts/train_smc_dl_v2.py` for one symbol as a background "
    "subprocess and returns immediately with the run's id and `state='running'`. Poll "
    "`GET /model-training/runs/current` for progress. On success the run overwrites that "
    "symbol's `<model>.pt`, `<model>_scaler.npz` and `<model>_meta.json` in "
    "`data/ml_models/`, so the next strategy instantiation picks up the new weights. "
    "Nothing else is mutated: no strategy is activated, no position is touched, and no "
    "risk cap is read or written.",
    responses={
        409: {
            "description": "Another training run is already in flight. Two concurrent runs "
            "would race on the same weights/scaler pair and could leave a model that does "
            "not match its own scaler."
        },
        503: {"description": "This process was not built with a training service."},
    },
)
async def start_training(request: Request, body: StartTrainingRequest) -> TrainingRunOut:
    try:
        run = await _service(request).start(body.symbol)
    except TrainingAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return TrainingRunOut.from_domain(run)


@router.get(
    "/runs/current",
    response_model=TrainingRunOut | None,
    summary="Current or most recent training run",
    description="The run in flight, or the last one that finished, including its exit code "
    "and the tail of its stdout/stderr (which contains the walk-forward table). Returns "
    "`null` when no run has been started since this process booted — training history "
    "that survives a restart lives under `GET /model-training/models`.",
    responses={503: {"description": "This process was not built with a training service."}},
)
async def current_run(request: Request) -> TrainingRunOut | None:
    run = _service(request).current()
    return TrainingRunOut.from_domain(run) if run is not None else None


@router.get(
    "/runs",
    response_model=list[TrainingRunOut],
    summary="Recent training runs in this process",
    description="Up to the last ten finished runs, newest first. In-memory only — a backend "
    "restart clears it, because the durable record of what was trained is the model "
    "metadata under `GET /model-training/models`.",
    responses={503: {"description": "This process was not built with a training service."}},
)
async def list_runs(request: Request) -> list[TrainingRunOut]:
    return [TrainingRunOut.from_domain(run) for run in _service(request).recent_runs()]


@router.get(
    "/models",
    response_model=list[TrainedModelOut],
    summary="Trained models on disk, with their out-of-sample verdict",
    description="Every `*_meta.json` in `data/ml_models/`, newest first, with the walk-forward "
    "summary that decides whether the model is worth using: in-sample vs out-of-sample average "
    "R, how many folds were positive, Brier scores, and whether the gate threshold was found "
    "to be tunable at all. Win rate is deliberately absent — with a 0.2R trailing rule ~94% of "
    "trades close at exactly +0.20R, so it measures the trailing rule rather than the model.",
    responses={503: {"description": "This process was not built with a training service."}},
)
async def list_models(request: Request) -> list[TrainedModelOut]:
    return [TrainedModelOut.from_domain(model) for model in _service(request).trained_models()]


@router.get(
    "/models/{model_name}/walk-forward",
    response_model=list[WalkForwardFoldOut],
    summary="Per-fold walk-forward detail for one model",
    description="The chronological folds behind a model's summary: each fold's out-of-sample "
    "window, trade count, sum R, average R, profit factor, max drawdown and calibration Brier, "
    "plus the in-sample economics for the same fold. Use this to check that a model's edge is "
    "consistent across time rather than carried by one lucky block.",
    responses={404: {"description": "No model with that name has metadata on disk."}},
)
async def walk_forward(
    request: Request,
    model_name: str = Path(description="Model file stem, e.g. `smc_dl_m5_v2`."),
) -> list[WalkForwardFoldOut]:
    folds = _service(request).walk_forward(model_name)
    if not folds:
        raise HTTPException(status_code=404, detail=f"no metadata for model {model_name!r}")
    return [WalkForwardFoldOut.from_meta(fold) for fold in folds]
