from __future__ import annotations
import argparse
import json

from .config import load_config_from_env
from .utils.logging import setup_logging, log_event, new_run_id
from .pipeline import run_claim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config_from_env()
    logger = setup_logging(
        name="LLM_explainable",
        level=cfg.log_level,
        log_dir=cfg.log_dir,
        json_logs=cfg.json_logs,
    )
    run_id = new_run_id()

    log_event(logger, "run_start", run_id=run_id, claim=args.claim)

    out = run_claim(
        args.claim,
        verifier_model_path=cfg.verifier.model_path,
        generator_model_name=cfg.generator.model,
        cache_dir=cfg.rag.cache_dir,
        use_cache=(not args.no_cache) and cfg.rag.use_cache,
        rag_kwargs={},  # later: pass google_pse cfg etc
    )

    log_event(logger, "run_done", run_id=run_id, verdict=out.verdict_status, label=out.label)

    print(json.dumps(out.__dict__, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
