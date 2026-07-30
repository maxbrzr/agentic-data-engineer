import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from .config import PipelineConfig
from .contracts import ModelConfig
from .metadata import CroissantBakerMetadataGenerator
from .pipeline import DataEngineeringPipeline
from .retrieval import EXAMPLE_DATASETS, DcatApHubRetriever, list_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one of the enabled DCAT-AP tabular datasets for ML."
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
        help="Process all three enabled examples sequentially.",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List enabled examples without downloading or running an agent.",
    )
    parser.add_argument(
        "--harness",
        default="opencode",
        help="Installed harness adapter to use (currently: opencode).",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("AGENT_MODEL_PROVIDER", "opencode"),
        help="Model provider ID passed unchanged to the selected harness.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_MODEL_ID", "deepseek-v4-flash-free"),
        help="Model ID passed unchanged to the selected harness.",
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
    max_continuations: int,
    provider_retries: int,
    retry_backoff_seconds: float,
):
    if name != "opencode":
        raise ValueError(
            f"No adapter is installed for harness {name!r}. "
            "Implement AgentHarness and inject it into DataEngineeringPipeline."
        )

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

    if not args.all and not args.example:
        parser.error("choose --example, --all, or --list-examples")

    workspace_root = args.workspace_root.expanduser().resolve()
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
        model=ModelConfig(provider_id=args.provider, model_id=args.model),
        force_download=args.force_download,
    )
    pipeline = DataEngineeringPipeline(
        retriever=DcatApHubRetriever(verbose=not args.quiet_download),
        harness=_create_harness(
            args.harness,
            args.opencode_url,
            max_continuations=args.opencode_max_continuations,
            provider_retries=args.opencode_provider_retries,
            retry_backoff_seconds=args.opencode_retry_backoff,
        ),
        metadata_generator=CroissantBakerMetadataGenerator(),
        config=config,
    )

    keys = tuple(EXAMPLE_DATASETS) if args.all else tuple(args.example)
    for key in keys:
        if sandbox_manager is not None:
            sandbox_manager.ensure(key)
        result = pipeline.run(key)
        print(
            f"{result.dataset.spec.key}: harness={result.agent.harness}, "
            f"run={result.agent.run_id}, output={result.agent.output_dir}, "
            f"metadata={result.metadata.path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
