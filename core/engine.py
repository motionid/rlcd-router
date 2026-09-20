"""Public inference API for the Apple Silicon MLX backend."""

try:
    from core.engine_mlx import MODEL_ID
    from core.engine_mlx import (
        get_engine,
        load_model_by_id,
        list_loaded_models,
        run_parallel_for_model,
        run_parallel_generation,
        run_rlcd_generation,
        run_text_generation,
        run_text_generation_default,
    )
except ImportError as exc:  # pragma: no cover - depends on host platform
    raise RuntimeError(
        "The bundled inference backend requires Apple Silicon with mlx and mlx-lm. "
        "Install requirements.txt or replace core.engine with another backend."
    ) from exc

USE_MLX = True

__all__ = [
    "get_engine",
    "run_parallel_generation",
    "run_rlcd_generation",
    "load_model_by_id",
    "list_loaded_models",
    "run_parallel_for_model",
    "run_text_generation",
    "run_text_generation_default",
    "MODEL_ID",
    "USE_MLX",
]
