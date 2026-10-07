"""One logging setup for the server and the CLI."""

import logging


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # HTTP client libraries log full request URLs at INFO; those can include secrets
    # (e.g. the email relay URL), so only their warnings are kept.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
