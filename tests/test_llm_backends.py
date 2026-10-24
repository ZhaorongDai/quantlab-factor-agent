"""The LLM backends: recording and replay through Campaign.run(), and each provider adapter.

Recording and replay run the real campaign on the synthetic store with
``tests/fake_llm.FakeLLM`` as the provider. The ``anthropic`` and
``openai_compatible`` adapters are tested offline against a local HTTP
server that answers like the provider; the ``claude_code`` adapter on the
command it builds. The live smoke tests call the real providers and run
only when their credentials are present.
"""

import http.server
import json
import os
import shutil
import subprocess
import threading

import pytest

from factor_agent.campaign import Campaign
from factor_agent.config import ROLES, CampaignConfig
from factor_agent.llm.base import Request, make_backend
from factor_agent.llm.log import LLMLog, RecordingBackend
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, Interrupted, round_of

SETTINGS = {"candidates": 5, "min_per_type": 1, "retries": 1, "max_rounds": 2}


def _propose(number, directions):
    planted = [candidate("intraday move", "alpha = close / open")] if number == 1 else []
    noise = [
        candidate(f"noise {number}.{slot}", f"alpha = BackRef(volume, {(5, 10, 20, 40, 60)[slot]}) / WindowedAvg(volume, {(20, 60)[number - 1]})")
        for slot in range(len(directions))
    ]
    return (planted + noise)[:len(directions)]


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    return {"components": synthetic_market.build(root / "data"), "cache": root / "kunquant_cache"}


@pytest.fixture(scope="module")
def recorded(tmp_path_factory, market):
    root = tmp_path_factory.mktemp("recorded")
    fake = FakeLLM(_propose, cost=0.25)
    result = run(write_config(root, market["components"], round=SETTINGS), fake, market["cache"])
    return {"root": root, "result": result, "fake": fake}


def _replay_roles(log) -> dict:
    return {role: {"backend": "replay", "model": f"model-{role}", "log": str(log)} for role in ROLES}


def test_every_call_is_recorded_in_full_under_the_run_llm_log(recorded):
    files = sorted((recorded["root"] / "run" / "llm_log").glob("*.json"))
    assert len(files) == len(recorded["fake"].requests) > 0
    first = json.loads(files[0].read_text())
    request = recorded["fake"].requests[0]
    assert first["request"] == {"role": request.role, "model": request.model, "prompt": request.prompt, "params": {}}
    assert first["reply"]["model_id"] == f"fake-{request.role}" and first["reply"]["cost"] == 0.25
    assert first["reply"]["text"].startswith("Here it is.")
    assert {"usage", "raw"} <= set(first["reply"])


def test_replaying_a_recorded_campaign_gives_an_identical_ledger(tmp_path, recorded, market):
    log = recorded["root"] / "run" / "llm_log"
    path = write_config(tmp_path, market["components"], round=SETTINGS, llm=_replay_roles(log))
    result = run(path, None, market["cache"])
    assert result.accepted, "the recorded campaign should accept the planted signal"
    original = recorded["root"] / "run"
    assert (tmp_path / "run" / "ledger.jsonl").read_text() == (original / "ledger.jsonl").read_text()
    assert result.spent == recorded["result"].spent == 0.25 * len(recorded["fake"].requests)
    replayed = {p.name: p.read_text() for p in (tmp_path / "run" / "llm_log").glob("*.json")}
    assert replayed == {p.name: p.read_text() for p in log.glob("*.json")}


def test_a_replay_that_runs_out_of_recorded_calls_fails_clearly(tmp_path, recorded, market):
    from factor_agent.llm.replay import ReplayExhausted

    log = tmp_path / "short_log"
    log.mkdir()
    files = sorted((recorded["root"] / "run" / "llm_log").glob("*.json"))
    for file in files[:-1]:
        (log / file.name).write_text(file.read_text())
    path = write_config(tmp_path, market["components"], round=SETTINGS, llm=_replay_roles(log))
    with pytest.raises(ReplayExhausted, match=r"no recorded reply left for this reviewer request"):
        run(path, None, market["cache"])


def _request(role="proposer", prompt="Write one factor.") -> Request:
    return Request(role=role, model="unused", prompt=prompt)


def test_the_claude_code_adapter_disables_every_tool():
    backend = make_backend({"backend": "claude_code", "model": "claude-opus-5-5"})
    command = backend.command()
    assert command[:2] == ["claude", "-p"]
    assert command[command.index("--tools") + 1] == ""  # no built-in tool
    assert command[command.index("--mcp-config") + 1] == '{"mcpServers": {}}' and "--strict-mcp-config" in command
    assert "--disable-slash-commands" in command
    assert command[command.index("--setting-sources") + 1] == ""  # no settings that could grant tools or hooks
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--model") + 1] == "claude-opus-5-5"


def test_the_claude_code_adapter_reads_the_json_reply_and_uses_the_local_login(tmp_path, monkeypatch):
    seen = tmp_path / "seen"
    fake_claude = tmp_path / "claude"
    reply = {
        "type": "result", "is_error": False, "result": "the answer", "total_cost_usd": 0.0125,
        "usage": {"input_tokens": 120, "output_tokens": 30}, "modelUsage": {"claude-opus-5-5": {"costUSD": 0.0125}},
    }
    fake_claude.write_text(
        "#!/bin/sh\n"
        f"cat > {seen}.stdin\n"
        f"echo \"key=${{ANTHROPIC_API_KEY:-none}} cwd=$(pwd)\" > {seen}.env\n"
        f"cat <<'JSON'\n{json.dumps(reply)}\nJSON\n"
    )
    fake_claude.chmod(0o755)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-sentinel")
    backend = make_backend({"backend": "claude_code", "model": "claude-opus-5-5", "executable": str(fake_claude)})
    answer = backend.complete(_request())
    assert (answer.text, answer.cost, answer.model_id) == ("the answer", 0.0125, "claude-opus-5-5")
    assert answer.usage == {"input_tokens": 120, "output_tokens": 30}
    assert json.loads(answer.raw) == reply
    assert seen.with_suffix(".stdin").read_text() == "Write one factor."
    env = seen.with_suffix(".env").read_text()
    assert "key=none" in env  # the API key is withheld: the call goes through the Claude Code login
    assert str(tmp_path) not in env  # run from an empty directory, away from any project


class _Provider:
    """A local HTTP server answering every POST with ``reply`` and keeping what it was sent."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.received: list[dict] = []
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                provider.received.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": json.loads(body)})
                data = json.dumps(provider.reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def provider():
    servers = []

    def start(reply):
        servers.append(_Provider(reply))
        return servers[-1]
    yield start
    for server in servers:
        server.server.shutdown()


ANTHROPIC_REPLY = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "the answer"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 1000, "output_tokens": 500},
}
OPENAI_REPLY = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "deepseek-chat",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "the answer"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
}


def test_the_anthropic_adapter_reads_its_key_from_the_environment_and_prices_the_usage(provider, monkeypatch):
    server = provider(ANTHROPIC_REPLY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-sentinel")
    backend = make_backend({
        "backend": "anthropic", "model": "claude-opus-5-5", "base_url": server.url, "max_tokens": 4000,
        "effort": "high", "input_usd_per_mtok": 4.0, "output_usd_per_mtok": 20.0,
    })
    answer = backend.complete(_request())
    (sent,) = server.received
    sent_key = sent["headers"].get("x-api-key") or sent["headers"]["authorization"].removeprefix("Bearer ")
    assert sent_key == "sk-secret-sentinel"
    assert sent["body"]["model"] == "claude-opus-5-5" and sent["body"]["max_tokens"] == 4000
    assert sent["body"]["messages"] == [{"role": "user", "content": "Write one factor."}]
    assert sent["body"]["output_config"] == {"effort": "high"}
    assert answer.text == "the answer" and answer.model_id == "claude-opus-5-5"
    assert answer.cost == pytest.approx(1000 * 4.0 / 1e6 + 500 * 20.0 / 1e6)
    assert answer.params == {"max_tokens": 4000, "effort": "high"}


def test_the_openai_compatible_adapter_reads_the_key_variable_the_config_names(provider, monkeypatch):
    server = provider(OPENAI_REPLY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-sentinel")
    backend = make_backend({
        "backend": "openai_compatible", "model": "deepseek-chat", "base_url": server.url + "/v1",
        "api_key_env": "DEEPSEEK_API_KEY", "max_tokens": 4000, "input_usd_per_mtok": 0.5, "output_usd_per_mtok": 2.0,
    })
    answer = backend.complete(_request())
    (sent,) = server.received
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer sk-secret-sentinel"
    assert sent["body"]["messages"] == [{"role": "user", "content": "Write one factor."}]
    assert answer.text == "the answer" and answer.model_id == "deepseek-chat"
    assert answer.cost == pytest.approx(1000 * 0.5 / 1e6 + 500 * 2.0 / 1e6)


def test_a_missing_key_variable_is_named(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_backend({"backend": "anthropic", "model": "m", "input_usd_per_mtok": 1, "output_usd_per_mtok": 1})
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(ValueError, match="NO_SUCH_KEY"):
        make_backend({
            "backend": "openai_compatible", "model": "m", "base_url": "http://127.0.0.1:1/v1",
            "api_key_env": "NO_SUCH_KEY", "input_usd_per_mtok": 1, "output_usd_per_mtok": 1,
        })


@pytest.mark.parametrize("option", ["api_key", "key", "token", "secret", "ANTHROPIC_API_KEY"])
def test_a_key_in_the_config_is_refused(option):
    with pytest.raises(ValueError, match="environment variable"):
        make_backend({
            "backend": "openai_compatible", "model": "m", "base_url": "http://127.0.0.1:1/v1",
            "input_usd_per_mtok": 1, "output_usd_per_mtok": 1, option: "sk-secret-sentinel",
        })


def test_no_key_appears_in_the_llm_log(tmp_path, provider, monkeypatch):
    anthropic_server, openai_server = provider(ANTHROPIC_REPLY), provider(OPENAI_REPLY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-sentinel-1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-sentinel-2")
    log = LLMLog(tmp_path / "llm_log")
    prices = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 1.0}
    for spec in (
        {"backend": "anthropic", "model": "claude-opus-5-5", "base_url": anthropic_server.url, **prices},
        {"backend": "openai_compatible", "model": "deepseek-chat", "base_url": openai_server.url + "/v1",
         "api_key_env": "DEEPSEEK_API_KEY", **prices},
    ):
        RecordingBackend(make_backend(spec), log).complete(_request())
    written = "".join(file.read_text() for file in log.files())
    assert len(log.files()) == 2 and "the answer" in written
    assert "sk-secret-sentinel" not in written


def _anthropic_roles(**options) -> dict:
    return {
        role: {"backend": "anthropic", "model": "claude-opus-5-5", "input_usd_per_mtok": 4.0,
               "output_usd_per_mtok": 20.0, **options}
        for role in ROLES
    }


def test_the_cost_estimate_grows_with_the_rounds_and_counts_every_call(tmp_path, market, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-unused")
    estimates = {}
    for rounds in (5, 10):
        path = write_config(tmp_path / str(rounds), market["components"], llm=_anthropic_roles(),
                            round={**SETTINGS, "max_rounds": rounds, "memory_every": 5})
        estimates[rounds] = Campaign(CampaignConfig.from_file(path)).estimate()
    five, ten = estimates[5], estimates[10]
    # Per round: a plan, a proposal and one fix, a review; a distillation every five rounds after the first five.
    assert five.calls == {"planner": 5, "proposer": 10, "reviewer": 5, "distiller": 0}
    assert ten.calls == {"planner": 10, "proposer": 20, "reviewer": 10, "distiller": 1}
    assert ten.total > 2 * five.total > 0  # the history every prompt carries grows with the rounds


def test_the_cost_estimate_is_printed_before_the_run(tmp_path, market, capsys):
    fake = FakeLLM(_propose)
    run(write_config(tmp_path, market["components"], round={**SETTINGS, "max_rounds": 1, "budget_usd": 2.5}),
        fake, market["cache"])
    out = capsys.readouterr().out
    assert "Estimated LLM cost" in out and "4 calls" in out and "budget $2.50" in out
    assert "unknown" in out  # the fake cannot price its calls


def test_an_estimate_above_the_budget_is_flagged(tmp_path, market, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-unused")
    path = write_config(tmp_path, market["components"], llm=_anthropic_roles(), round={**SETTINGS, "budget_usd": 0.01})
    summary = Campaign(CampaignConfig.from_file(path)).estimate().summary(0.01)
    assert "budget $0.01" in summary and "the estimate is above the budget" in summary


# Live smoke tests: one real call each, skipped without the provider's credentials.

SMOKE_PROMPT = "Reply with the single word: ready"


def _claude_code_logged_in() -> bool:
    if shutil.which("claude") is None:
        return False
    status = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True)
    return status.returncode == 0 and '"loggedIn": true' in status.stdout


@pytest.mark.skipif(not _claude_code_logged_in(), reason="no Claude Code login")
def test_live_claude_code():
    answer = make_backend({"backend": "claude_code", "model": "claude-opus-5-5", "effort": "low"}).complete(
        _request(prompt=SMOKE_PROMPT))
    assert "ready" in answer.text.lower() and answer.cost > 0 and answer.model_id


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
def test_live_anthropic():
    answer = make_backend({
        "backend": "anthropic", "model": "claude-opus-5-5", "effort": "low", "max_tokens": 1024,
        "input_usd_per_mtok": 4.0, "output_usd_per_mtok": 20.0,
    }).complete(_request(prompt=SMOKE_PROMPT))
    assert "ready" in answer.text.lower() and answer.cost > 0 and answer.model_id.startswith("claude-opus-5-5")


#: The OpenAI-compatible provider of the smoke test: FACTOR_AGENT_SMOKE_OPENAI_BASE_URL, _MODEL and
#: _KEY_ENV (the name of the variable holding its key), or DeepSeek when DEEPSEEK_API_KEY is set.
_OPENAI_SMOKE = {
    "base_url": os.environ.get("FACTOR_AGENT_SMOKE_OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    "model": os.environ.get("FACTOR_AGENT_SMOKE_OPENAI_MODEL", "deepseek-chat"),
    "api_key_env": os.environ.get("FACTOR_AGENT_SMOKE_OPENAI_KEY_ENV", "DEEPSEEK_API_KEY"),
}


@pytest.mark.skipif(not os.environ.get(_OPENAI_SMOKE["api_key_env"]), reason=f"{_OPENAI_SMOKE['api_key_env']} is not set")
def test_live_openai_compatible():
    answer = make_backend({
        "backend": "openai_compatible", **_OPENAI_SMOKE, "max_tokens": 64,
        "input_usd_per_mtok": 1.0, "output_usd_per_mtok": 1.0,
    }).complete(_request(prompt=SMOKE_PROMPT))
    assert "ready" in answer.text.lower() and answer.cost > 0 and answer.model_id


def test_a_resumed_run_replays_with_the_replies_its_ledger_used(tmp_path, market):
    # The crash cuts round 2 off after its proposal; the resumed run proposes round 2 afresh,
    # differently. Replay must serve the second proposal, the one the ledger holds.
    root = tmp_path / "resumed"
    path = write_config(root, market["components"], round=SETTINGS)
    other = lambda number, directions: _propose(number, directions) if number == 1 else [
        candidate(f"other {slot}", f"alpha = BackRef(close, {(5, 10, 20, 40, 60)[slot]}) / WindowedAvg(close, 20)")
        for slot in range(len(directions))
    ]
    crashing = FakeLLM(other, interrupt=lambda r: r.role == "reviewer" and round_of(r.prompt) == 2)
    with pytest.raises(Interrupted):
        run(path, crashing, market["cache"])
    result = run(path, FakeLLM(_propose), market["cache"])
    assert result.rounds == 2
    dropped = list((root / "run" / "llm_log" / "dropped").glob("*.json"))
    assert {json.loads(f.read_text())["request"]["role"] for f in dropped} == {"planner", "proposer"}

    replayed = tmp_path / "replayed"
    run(write_config(replayed, market["components"], round=SETTINGS, llm=_replay_roles(root / "run" / "llm_log")),
        None, market["cache"])
    ledger = (root / "run" / "ledger.jsonl").read_text()
    assert "noise 2.0" in ledger and "other 0" not in ledger
    assert (replayed / "run" / "ledger.jsonl").read_text() == ledger


def test_a_replay_of_a_missing_log_says_so(tmp_path):
    with pytest.raises(ValueError, match="no recorded call in .*nowhere"):
        make_backend({"backend": "replay", "model": "m", "log": str(tmp_path / "nowhere")})
