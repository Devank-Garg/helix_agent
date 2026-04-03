"""helix_agent.cli.main — Entry point, argument parsing, and top-level run()."""

import sys
import click
import helix_agent


@click.group(invoke_without_command=True)
@click.version_option(version=helix_agent.__version__, prog_name="helix-agent")
@click.pass_context
def main(ctx: click.Context) -> None:
    """helix-agent — AI coding assistant powered by Claude."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("prompt", required=False)
@click.option("-p", "--prompt", "prompt_opt", default=None, help="Prompt to send.")
@click.option(
    "--model",
    default="sonnet",
    show_default=True,
    help="Model alias: opus | sonnet | haiku, or a full model ID.",
)
@click.option(
    "--dangerously-skip-permissions",
    is_flag=True,
    default=False,
    help="Skip all permission checks (DangerFullAccess mode).",
)
def chat(
    prompt: str | None,
    prompt_opt: str | None,
    model: str,
    dangerously_skip_permissions: bool,
) -> None:
    """Start an interactive chat session or send a single prompt."""
    # Phase 2+ will wire up the real conversation runtime here.
    effective_prompt = prompt_opt or prompt
    click.echo(
        f"[stub] model={model!r}  skip_permissions={dangerously_skip_permissions}"
    )
    if effective_prompt:
        click.echo(f"[stub] prompt={effective_prompt!r}")
    else:
        click.echo("[stub] entering interactive REPL (not yet implemented)")


if __name__ == "__main__":
    main()
