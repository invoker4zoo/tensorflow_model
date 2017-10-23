
import logging
import logging.config
from logging_conf import BORN_LOGGING_CONF

logging.config.dictConfig(BORN_LOGGING_CONF)
logger = logging.getLogger("born")
# logger.info('testing1')