from importlib.metadata import version

import factor_agent


def test_version():
    assert factor_agent.__version__ == version("quantlab-factor-agent")
