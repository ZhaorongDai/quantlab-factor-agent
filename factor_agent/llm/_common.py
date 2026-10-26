"""What the provider adapters share: the key from its environment variable."""

import os


def key_from_environment(name: str) -> str:
    """The value of the environment variable ``name``.

    Raises
    ------
    ValueError
        If the variable is unset or empty.
    """
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"the environment variable {name} holding the API key is not set")
    return value

