from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from . import __version__, config, db, engine, indexer

_STAGE = {
    "text": ("📝", "Indexing"),
    "pdf": ("📄", "Reading PDF"),
    "html": ("🌐", "Reading HTML"),
    "docx": ("📄", "Reading doc"),
    "image": ("🖼 ", "Image"),
    "ocr": ("🔤", "OCR"),
    "audio": ("🎵", "Transcribing"),
    "video": ("🎬", "Transcribing"),
    "transcribe": ("🎬", "Transcribing"),
    "embed": ("🧠", "Embedding"),
}


def quiet_noisy_logs() -> None:
    import logging
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in (
        "pypdf",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "transformers",
        "sentence_transformers",
        "PIL",
        "onnxruntime",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def _fmt_ts(ms: int | None) -> str:
    if ms is None:
        return ""
    s = ms // 1000
    return f" @ {s // 60:d}:{s % 60:02d}"


def cmd_init(args) -> int:
    path, created = config.write_default_config()
    if created:
        print(f"Wrote default config to {path}")
    else:
        print(f"Config already exists at {path}, leaving it alone")
    print(f"Index DB will live at {config.db_path()}")
    return 0


def cmd_reindex(args) -> int:
    cfg = config.load()
    roots = [Path(p).expanduser() for p in args.paths] if args.paths else None
    if roots is None and (not cfg.folders):
        print(
            "No folders configured. Edit",
            config.config_path(),
            "or pass paths: sift reindex ~/Documents",
            file=sys.stderr,
        )
        return 2
    embedder = engine.build_embedder(cfg)
    image_embedder = engine.build_image_embedder(cfg)
    transcriber = engine.build_transcriber(cfg)
    ocr_engine = engine.build_ocr_engine(cfg)
    con = db.connect(config.db_path(), load_vec=embedder is not None or image_embedder is not None)
    if not db.has_trigram(con):
        print("warning: SQLite lacks the trigram tokenizer; fuzzy match disabled.", file=sys.stderr)
    enabled = []
    if embedder is not None:
        enabled.append(f"semantic({cfg.text_model})")
    if image_embedder is not None:
        enabled.append(f"image({cfg.image_model})")
    if transcriber is not None:
        enabled.append(f"asr({cfg.asr_model})")
    if ocr_engine is not None:
        enabled.append(f"ocr({cfg.ocr_engine})")
    dev = getattr(embedder or image_embedder or transcriber, "device", "cpu")
    print(f"Indexing with: {', '.join(enabled) or 'full text only'}  [device={dev}]")
    res = _reindex_with_progress(con, cfg, roots, embedder, image_embedder, transcriber, ocr_engine)
    print(
        f"\n✓ scanned={res.scanned} indexed={res.indexed} skipped={res.skipped} deferred={res.deferred} errors={res.errors}"
    )
    return 0


def _reindex_with_progress(con, cfg, roots, embedder, image_embedder, transcriber, ocr_engine):
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        return indexer.reindex(con, cfg, roots, embedder, image_embedder, transcriber, ocr_engine)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as prog:
        task = prog.add_task("Scanning…", total=None)

        def cb(done, total, path, stage):
            if stage == "done":
                prog.update(task, total=total, completed=done)
                return
            icon, verb = _STAGE.get(stage, ("•", stage))
            name = os.path.basename(str(path))
            prog.update(task, total=total, description=f"{icon} {verb}: [dim]{name}[/dim]")

        return indexer.reindex(
            con, cfg, roots, embedder, image_embedder, transcriber, ocr_engine, progress=cb
        )


def cmd_search(args) -> int:
    cfg = config.load()
    eng = engine.SearchEngine(
        embedder=None if args.no_semantic else engine.build_embedder(cfg),
        image_embedder=None if args.no_images else engine.build_image_embedder(cfg),
        cfg=cfg,
    )
    hits = eng.search(args.query, limit=args.limit, path_prefix=args.path)
    if args.json:
        import dataclasses
        import json

        payload = {
            "query": args.query,
            "count": len(hits),
            "hits": [dataclasses.asdict(h) for h in hits],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("No results.")
        return 0
    for i, h in enumerate(hits, 1):
        loc = _fmt_ts(h.start_ms) or (f" (p.{h.page})" if h.page else "")
        print(f"{i:2d}. [{h.kind}]{loc} {h.path}")
        if h.snippet:
            print(f"    {h.snippet}")
    return 0


def cmd_status(args) -> int:
    con = db.connect(config.db_path())
    s = db.stats(con)
    print(f"DB: {config.db_path()}")
    print(f"files={s['files']} chunks={s['chunks']}")
    print(f"by status: {s['by_status']}")
    print(f"by kind:   {s['by_kind']}")
    return 0


def cmd_gui(args) -> int:
    from .gui.app import run_gui

    return run_gui(initial_query=args.query, path=args.path)


def cmd_krunner(args) -> int:
    from .krunner import run_krunner_service

    return run_krunner_service()


def cmd_install_kde(args) -> int:
    from .kde_install import install_kde

    install_kde()
    return 0


def cmd_bench(args) -> int:
    from .bench import run_bench

    return run_bench(n_docs=args.docs, words=args.words, queries=args.queries)


def cmd_install_service(args) -> int:
    from .service_install import install_service

    install_service(enable=not args.no_enable)
    return 0


def cmd_config(args) -> int:
    cfg = config.load()
    print(f"config: {config.config_path()}")
    print(f"profile: {cfg.profile}  features: {cfg.features}")
    print(f"folders: {cfg.folders or '(none configured)'}")
    return 0


def cmd_paths(args) -> int:
    from . import maint

    stats = maint.index_stats()
    print(f"config:  {config.config_path()}")
    print(f"data:    {config.data_dir()}")
    print(
        f"index:   {config.db_path()}  "
        f"({maint.human_size(maint.index_size())}, "
        f"{stats['files']} files, {stats['chunks']} chunks)"
    )
    print(f"models:  {maint.models_dir()}  ({maint.human_size(maint.models_total())})")
    for name, size in maint.models_listing():
        print(f"           {name}  {maint.human_size(size)}")
    return 0


def cmd_clear_index(args) -> int:
    from . import maint

    print(f"Cleared index, freed {maint.human_size(maint.clear_index())}.")
    return 0


def cmd_clear_models(args) -> int:
    from . import maint

    print(f"Cleared model downloads, freed {maint.human_size(maint.clear_models())}.")
    return 0


def cmd_download_models(args) -> int:
    from . import download

    download.download_enabled(config.load())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sift", description="Local file search for KDE.")
    p.add_argument("--version", action="version", version=f"sift {__version__}")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="show third party warnings and logs"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="write a default config file").set_defaults(func=cmd_init)
    r = sub.add_parser("reindex", help="index configured folders (or given paths)")
    r.add_argument("paths", nargs="*", help="optional paths to index")
    r.set_defaults(func=cmd_reindex)
    i = sub.add_parser("index", help="alias for reindex")
    i.add_argument("paths", nargs="*")
    i.set_defaults(func=cmd_reindex)
    s = sub.add_parser("search", help="search the index")
    s.add_argument("query")
    s.add_argument("-n", "--limit", type=int, default=20)
    s.add_argument("--path", default=None, help="scope results under this folder")
    s.add_argument("--json", action="store_true", help="emit results as JSON")
    s.add_argument("--no-semantic", action="store_true", help="skip text vector search")
    s.add_argument("--no-images", action="store_true", help="skip text->image search")
    s.set_defaults(func=cmd_search)
    g = sub.add_parser("gui", help="open the search window")
    g.add_argument("query", nargs="?", default="", help="initial query")
    g.add_argument("--path", default=None, help="scope results under this folder")
    g.set_defaults(func=cmd_gui)
    sub.add_parser("krunner", help="run the KRunner D-Bus service").set_defaults(func=cmd_krunner)
    sub.add_parser("install-kde", help="install KRunner + Dolphin integration").set_defaults(
        func=cmd_install_kde
    )
    iv = sub.add_parser("install-service", help="install the daily reindex systemd timer")
    iv.add_argument("--no-enable", action="store_true", help="write units but don't enable")
    iv.set_defaults(func=cmd_install_service)
    b = sub.add_parser("bench", help="benchmark throughput + desktop impact")
    b.add_argument("--docs", type=int, default=800)
    b.add_argument("--words", type=int, default=400)
    b.add_argument("--queries", type=int, default=50)
    b.set_defaults(func=cmd_bench)
    sub.add_parser("status", help="show index stats").set_defaults(func=cmd_status)
    sub.add_parser("config", help="show effective config").set_defaults(func=cmd_config)
    sub.add_parser("paths", help="show config/index/model locations and sizes").set_defaults(
        func=cmd_paths
    )
    sub.add_parser("clear-index", help="delete the search index").set_defaults(func=cmd_clear_index)
    sub.add_parser("clear-models", help="delete downloaded models").set_defaults(
        func=cmd_clear_models
    )
    sub.add_parser("download-models", help="download models ahead of first use").set_defaults(
        func=cmd_download_models
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "verbose", False):
        quiet_noisy_logs()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
