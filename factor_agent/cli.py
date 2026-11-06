"""The ``factor-agent`` console script: ``factor-agent check <config> [--ping]``.

``check`` prints the report of ``Campaign.check`` and exits with status 1
when it holds a problem, 0 otherwise. A config that cannot be read is the
report's first and only problem, printed like any other, not a traceback.
"""

import argparse
import sys
from pathlib import Path

from factor_agent.preflight import Check, CheckReport


def check(path: Path, ping: bool = False) -> CheckReport:
    """The report of the campaign config at ``path``: ``Campaign.check``'s, or the problem of reading the config."""
    from factor_agent.campaign import Campaign
    from factor_agent.config import CampaignConfig

    try:
        config = CampaignConfig.from_file(path)
    except Exception as exc:  # a config that cannot be read is reported, not raised
        return CheckReport((Check("config", problems=(f"{path}: {type(exc).__name__}: {exc}",)),))
    return Campaign(config).check(ping=ping)


def main(argv: list[str] | None = None) -> int:
    """Run the ``factor-agent`` console script; return its exit status."""
    parser = argparse.ArgumentParser(prog="factor-agent", description="Work with campaign configs.")
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser(
        "check", help="check a campaign config and its run directory without starting the run",
        description="Report everything a run of the config would refuse before its first LLM call, writing "
                    "nothing; exit 1 if there is a problem.",
    )
    checking.add_argument("config", type=Path, help="the campaign config (YAML)")
    checking.add_argument(
        "--ping", action="store_true",
        help="also send each distinct LLM backend one short request (replay and scripted roles are not pinged)",
    )
    args = parser.parse_args(argv)
    report = check(args.config, ping=args.ping)
    print(report.text(), flush=True)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
