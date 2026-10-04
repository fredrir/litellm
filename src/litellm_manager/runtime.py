"""Load the user's dotenv configuration at service startup, without copying credentials."""

import os
import sys

from .settings import Settings


def main():
    settings = Settings()
    env = {**os.environ, "LITELLM_MASTER_KEY": settings.require_key()}
    os.execve(sys.argv[1], sys.argv[1:], env)


if __name__ == "__main__":
    main()
