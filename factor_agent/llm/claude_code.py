"""The Claude Code backend: ``claude -p`` headless, JSON output, every tool off, the local login.

Every built-in tool, MCP server, skill, slash command and settings file is
turned off and the call runs from an empty directory, so the model only
writes text and cannot read the data (Q26). ``ANTHROPIC_API_KEY`` and
``ANTHROPIC_AUTH_TOKEN`` are kept out of the child's environment, so the
call goes through the Claude Code login. Managed (enterprise) settings
cannot be turned off from the command line. Claude Code reports each call's cost itself.
"""

import json
import os
import subprocess
import tempfile

from factor_agent.llm._common import Prices
from factor_agent.llm.base import Reply, Request


class ClaudeCodeBackend:
    """Complete a request with one headless Claude Code call.

    Parameters
    ----------
    model : str
        The ``--model`` of every call.
    effort : str, optional
        The ``--effort`` of every call; Claude Code's default if None.
    executable : str
        The Claude Code command.
    timeout_seconds : float
        Seconds a call may take.
    prices : Prices, optional
        What the model's tokens cost, for the estimate before a run only:
        Claude Code reports each call's actual cost. None: the estimate is unknown.
    """

    def __init__(
        self, model: str, effort: str | None = None, executable: str = "claude", timeout_seconds: float = 900.0,
        prices: Prices | None = None,
    ):
        self.model = model
        self.prices = prices
        self.effort = effort
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, spec: dict) -> "ClaudeCodeBackend":
        """Build from a role's config: ``model`` and optionally ``effort``, ``executable``, ``timeout_seconds`` and the prices."""
        options = {k: spec[k] for k in ("effort", "executable", "timeout_seconds") if k in spec}
        if "input_usd_per_mtok" in spec or "output_usd_per_mtok" in spec:
            options["prices"] = Prices.from_spec(spec, "claude_code")
        return cls(spec["model"], **options)

    def estimate(self, input_tokens: int, output_tokens: int) -> float | None:
        """What a call with these token counts would cost, or None without prices in the config."""
        return self.prices.cost(input_tokens, output_tokens) if self.prices else None

    def command(self) -> list[str]:
        """The command line of every call; the prompt goes on stdin."""
        command = [
            self.executable, "-p", "--output-format", "json", "--model", self.model,
            "--tools", "", "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}),
            "--disable-slash-commands", "--setting-sources", "", "--no-session-persistence",
        ]
        if self.effort:
            command += ["--effort", self.effort]
        return command

    def complete(self, request: Request) -> Reply:
        """Run the call and read its JSON reply.

        Raises
        ------
        RuntimeError
            If Claude Code exits with an error, reports one, or reports no cost.
        subprocess.TimeoutExpired
            If the call takes longer than ``timeout_seconds``.
        """
        env = {k: v for k, v in os.environ.items() if k not in _CREDENTIALS}
        with tempfile.TemporaryDirectory(prefix="factor_agent_claude_") as empty:
            finished = subprocess.run(
                self.command(), input=request.prompt, capture_output=True, text=True, cwd=empty, env=env,
                timeout=self.timeout_seconds,
            )
        if finished.returncode != 0:
            raise RuntimeError(f"claude_code: exit {finished.returncode}: {(finished.stderr or finished.stdout)[-2000:]}")
        reply = json.loads(finished.stdout)
        if reply.get("is_error"):
            raise RuntimeError(f"claude_code: the call failed: {str(reply.get('result'))[:2000]}")
        if not isinstance(reply.get("total_cost_usd"), (int, float)):
            raise RuntimeError("claude_code: the reply reports no total_cost_usd, so the budget cannot count it")
        usage = reply.get("usage") or {}
        models = reply.get("modelUsage") or {}
        return Reply(
            text=reply["result"],
            usage={k: usage[k] for k in ("input_tokens", "output_tokens") if k in usage},
            cost=float(reply["total_cost_usd"]),
            model_id=self.model if self.model in models else next(iter(models), self.model),
            params={"effort": self.effort} if self.effort else {},
            raw=finished.stdout.strip(),
        )


#: Environment variables kept from the child, so it answers through the Claude Code login.
_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
