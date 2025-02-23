import logging
import os
import sys

# Define standard log levels
log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

# Set global log level from environment variable
GLOBAL_LOG_LEVEL = os.environ.get("GLOBAL_LOG_LEVEL", "").upper()
if GLOBAL_LOG_LEVEL in log_levels:
    logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL, force=True)
else:
    GLOBAL_LOG_LEVEL = "INFO"

log = logging.getLogger(__name__)
log.info(f"GLOBAL_LOG_LEVEL: {GLOBAL_LOG_LEVEL}")

# Define custom log sources
log_sources = ["CONFIG", "MODEL", "RAG", "ROUTER", "SERVICE", "UTILS"]

# Store log levels for each source
SRC_LOG_LEVELS = {}

for source in log_sources:
    log_env_var = source + "_LOG_LEVEL"
    SRC_LOG_LEVELS[source] = os.environ.get(log_env_var, "").upper()
    if SRC_LOG_LEVELS[source] not in log_levels:
        SRC_LOG_LEVELS[source] = GLOBAL_LOG_LEVEL
    log.info(f"{log_env_var}: {SRC_LOG_LEVELS[source]}")

# Example usage: Set log level for CONFIG source
log.setLevel(SRC_LOG_LEVELS["CONFIG"])
