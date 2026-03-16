"""Logging setup for Woven Imprint."""

import logging

logger = logging.getLogger("woven_imprint")

# Don't add handlers by default — let the application configure logging.
# Users can enable debug logging with:
#   logging.basicConfig(level=logging.DEBUG)
#   or
#   logging.getLogger("woven_imprint").setLevel(logging.DEBUG)
