import logging

from .gades import GADESForceUpdater, createGADESBiasForce, GADESBias


class ColorFormatter(logging.Formatter):
    """Custom logging formatter with ANSI color codes for terminal output."""

    COLORS = {
        'DEBUG': '\033[0;36m',     # Cyan
        'INFO': '\033[1;32m',      # Bold Green
        'WARNING': '\033[1;33m',   # Bold Yellow
        'ERROR': '\033[1;31m',     # Bold Red
        'CRITICAL': '\033[1;35m',  # Bold Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


# Create and configure GADES logger
logger = logging.getLogger("GADES")
logger.setLevel(logging.INFO)

# Only add handler if none exist (avoid duplicates on reimport)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter('[GADES | %(levelname)s] %(message)s'))
    logger.addHandler(handler)


__all__ = [
    "GADESForceUpdater",
    "createGADESBiasForce",
    "GADESBias",
    "logger",
    "ColorFormatter",
]