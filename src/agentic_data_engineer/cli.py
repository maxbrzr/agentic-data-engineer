import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .agent.models import (
    DEFAULT_HARNESS,
    DEFAULT_MODEL_BY_PROVIDER,
    DEFAULT_PROVIDER,
    HARNESSES,
    MODELS,
    MODELS_BY_PROVIDER,
    PROVIDERS,
    PROVIDERS_BY_HARNESS,
    resolve_model_config,
)
from .config import PipelineConfig
from .metadata import CroissantBakerMetadataGenerator
from .pipeline import DataEngineeringPipeline
from .processor_check import DockerProcessorChecker
from .retrieval import (
    EXAMPLE_DATASETS,
    DcatApHubRetriever,
    dataset_spec_from_url,
    get_example,
    list_examples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a DCAT-AP tabular dataset for ML."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--example",
        action="append",
        choices=tuple(EXAMPLE_DATASETS),
        help="Example key to process. Repeat to process multiple examples.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process all enabled examples sequentially.",
    )
    selection.add_argument(
        "--url",
        help="Process one dataset from an absolute DCAT-AP HTTP(S) URL.",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List enabled examples without downloading or running an agent.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported providers and models per harness, then exit.",
    )
    parser.add_argument(
        "--harness",
        choices=HARNESSES,
        default=DEFAULT_HARNESS,
        help=f"Agent harness to run in Docker. Default: {DEFAULT_HARNESS}.",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=os.getenv("AGENT_MODEL_PROVIDER"),
        help=(
            "Model provider (pi supports gwdg and kit only). Defaults to "
            f"$AGENT_MODEL_PROVIDER, else {DEFAULT_PROVIDER}."
        ),
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        metavar="MODEL",
        default=os.getenv("AGENT_MODEL_ID"),
        help=(
            "Model ID; must belong to the provider (see --list-models). "
            "Defaults to $AGENT_MODEL_ID, else "
            + ", ".join(f"{m} ({p})" for p, m in DEFAULT_MODEL_BY_PROVIDER.items())
            + "."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace containing data/, output/, and the agent prompt.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Agent prompt path. Defaults to prompts/data-engineer.md in the workspace.",
    )
    parser.add_argument(
        "--opencode-url",
        default=os.getenv("OPENCODE_BASE_URL", "http://127.0.0.1:54321"),
        help="Local Docker OpenCode endpoint; used only by the OpenCode adapter.",
    )
    parser.add_argument(
        "--opencode-max-continuations",
        type=int,
        default=int(os.getenv("OPENCODE_MAX_CONTINUATIONS", "2")),
        help="Maximum follow-up turns after OpenCode becomes idle with missing outputs.",
    )
    parser.add_argument(
        "--opencode-provider-retries",
        type=int,
        default=int(os.getenv("OPENCODE_PROVIDER_RETRIES", "2")),
        help="Maximum retries for transient model-provider failures per turn.",
    )
    parser.add_argument(
        "--opencode-retry-backoff",
        type=float,
        default=float(os.getenv("OPENCODE_RETRY_BACKOFF_SECONDS", "2")),
        help="Initial provider-retry backoff in seconds; subsequent retries double it.",
    )
    parser.add_argument(
        "--no-manage-opencode-sandbox",
        action="store_true",
        help=(
            "Do not automatically start or switch the local OpenCode Compose "
            "sandbox. Mount verification remains enabled."
        ),
    )
    parser.add_argument(
        "--pi-max-continuations",
        type=int,
        default=int(os.getenv("PI_MAX_CONTINUATIONS", "2")),
        help="Maximum follow-up Pi invocations when required outputs are missing.",
    )
    parser.add_argument(
        "--pi-provider-retries",
        type=int,
        default=int(os.getenv("PI_PROVIDER_RETRIES", "2")),
        help="Maximum retries for transient Pi model-provider failures per turn.",
    )
    parser.add_argument(
        "--pi-retry-backoff",
        type=float,
        default=float(os.getenv("PI_RETRY_BACKOFF_SECONDS", "2")),
        help="Initial Pi provider-retry backoff in seconds.",
    )
    parser.add_argument(
        "--pi-timeout",
        type=float,
        default=float(os.getenv("PI_TIMEOUT_SECONDS", "1200")),
        help="Timeout in seconds for each Pi container invocation.",
    )
    parser.add_argument(
        "--pi-stall-timeout",
        type=float,
        default=float(os.getenv("PI_STALL_TIMEOUT_SECONDS", "300")),
        help="Abort and retry when Pi emits no output for this many seconds.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ask dcat-ap-hub to download files again.",
    )
    parser.add_argument(
        "--quiet-download",
        action="store_true",
        help="Disable dcat-ap-hub download progress output.",
    )
    return parser


def _create_harness(
    name: str,
    opencode_url: str,
    *,
    workspace_root: Path,
    max_continuations: int,
    provider_retries: int,
    retry_backoff_seconds: float,
    pi_max_continuations: int,
    pi_provider_retries: int,
    pi_retry_backoff_seconds: float,
    pi_timeout_seconds: float,
    pi_stall_timeout_seconds: float,
):
    if name == "pi":
        from .agent.pi import PiHarness, PiSettings

        return PiHarness(
            PiSettings(
                project_root=workspace_root,
                timeout_seconds=pi_timeout_seconds,
                stall_timeout_seconds=pi_stall_timeout_seconds,
                max_continuations=pi_max_continuations,
                max_provider_retries=pi_provider_retries,
                retry_backoff_seconds=pi_retry_backoff_seconds,
            )
        )
    if name != "opencode":
        raise ValueError(f"Unknown harness {name!r}")

    from .agent.opencode import OpencodeHarness, OpencodeSettings

    return OpencodeHarness(
        OpencodeSettings(
            base_url=opencode_url,
            max_continuations=max_continuations,
            max_provider_retries=provider_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            require_sandbox_preflight=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_examples:
        for spec in list_examples():
            print(f"{spec.key}: {spec.title}\n  {spec.url}")
        return 0

    if args.list_models:
        for harness, providers in PROVIDERS_BY_HARNESS.items():
            print(f"{harness}:")
            for provider in providers:
                print(f"  {provider}:")
                for model in MODELS_BY_PROVIDER[provider]:
                    default = (
                        " (default)"
                        if DEFAULT_MODEL_BY_PROVIDER[provider] == model
                        else ""
                    )
                    print(f"    {model}{default}")
        return 0

    if not args.all and not args.example and not args.url:
        parser.error("choose --example, --all, --url, or --list-examples")

    try:
        specs = (
            (dataset_spec_from_url(args.url),)
            if args.url
            else tuple(
                get_example(key)
                for key in (
                    tuple(EXAMPLE_DATASETS) if args.all else tuple(args.example)
                )
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    workspace_root = args.workspace_root.expanduser().resolve()
    try:
        model = resolve_model_config(args.harness, args.provider, args.model)
    except ValueError as exc:
        parser.error(str(exc))
    sandbox_manager = None
    if args.harness == "opencode":
        from .agent.opencode_sandbox import OpencodeSandboxManager

        try:
            configured_sandbox = OpencodeSandboxManager(
                project_root=workspace_root,
                base_url=args.opencode_url,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if not args.no_manage_opencode_sandbox:
            sandbox_manager = configured_sandbox

    prompt_path = args.prompt or workspace_root / "prompts" / "data-engineer.md"
    config = PipelineConfig(
        workspace_root=workspace_root,
        prompt_path=prompt_path,
        model=model,
        force_download=args.force_download,
    )
    pipeline = DataEngineeringPipeline(
        retriever=DcatApHubRetriever(verbose=not args.quiet_download),
        harness=_create_harness(
            args.harness,
            args.opencode_url,
            workspace_root=workspace_root,
            max_continuations=args.opencode_max_continuations,
            provider_retries=args.opencode_provider_retries,
            retry_backoff_seconds=args.opencode_retry_backoff,
            pi_max_continuations=args.pi_max_continuations,
            pi_provider_retries=args.pi_provider_retries,
            pi_retry_backoff_seconds=args.pi_retry_backoff,
            pi_timeout_seconds=args.pi_timeout,
            pi_stall_timeout_seconds=args.pi_stall_timeout,
        ),
        metadata_generator=CroissantBakerMetadataGenerator(),
        processor_checker=DockerProcessorChecker(workspace_root),
        config=config,
    )

    failed: list[str] = []
    for spec in specs:
        try:
            # Download and check the source before starting any agent.
            retrieved = pipeline.prepare(spec)
            if sandbox_manager is not None:
                sandbox_manager.ensure(
                    spec.key,
                    data_dir_name=spec.local_dir_name,
                    required_provider=model.provider_id,
                )
            result = pipeline.run_spec(spec, prepared=retrieved)
        except ValueError as exc:
            # Pipeline errors are explained in their message; no traceback.
            print(f"{spec.key}: error: {exc}", file=sys.stderr)
            failed.append(spec.key)
            continue
        print(
            f"{result.dataset.spec.key}: harness={result.agent.harness}, "
            f"run={result.agent.run_id}, output={result.agent.output_dir}, "
            f"metadata={result.metadata.path}, "
            f"provenance={result.provenance_path}"
        )
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
