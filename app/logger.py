from loguru import logger
from app.config import settings
import sys


# Clear default handlers
logger.remove()

logger.add(sys.stderr, level=settings.log_level, colorize=True,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                  "<level>{level: <8}</level> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                  "<level>{message}</level>")

logger.add(settings.log_file, rotation="500 KB", level=settings.log_level,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")