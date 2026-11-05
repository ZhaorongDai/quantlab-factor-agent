# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's
[private vulnerability reporting](https://github.com/ZhaorongDai/quantlab-factor-agent/security/advisories/new),
not in a public issue. Include what an attacker can do, the steps or a config that shows
it, and the version or commit you tested. You can expect a first reply within a week.

Only the latest release and `main` receive fixes.

## What the project protects

quantlab-factor-agent runs code that a language model writes. The Gate builds each
candidate's KunQuant graph in a separate subprocess run with `python -I`, an empty
environment, a scratch working directory, a timeout and a memory limit, and, where
[bubblewrap](https://github.com/containers/bubblewrap) is installed (Linux), a filesystem
view of its own; the code sees only the names of the vocabulary. A way for candidate code
to escape that subprocess, read files or the environment, or reach the network is a
vulnerability we want to hear about. Without bubblewrap the subprocess still shares the
user's filesystem, so run campaigns as a user that holds nothing you would not risk, on a
machine where bubblewrap is installed when you can.

Keys are read only from environment variables, and a config that holds one is refused. The
web frontend (`factor-agent-web`) is read-only and binds `127.0.0.1` only; reach it from
another machine through an SSH tunnel rather than by exposing the port.
