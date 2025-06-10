import logging
from logzero import setup_logger

logger = setup_logger(
    name='logger',
    level=logging.DEBUG,
    formatter=None,
    fileLoglevel=logging.DEBUG,
    disableStderrLogger=False,
    logfile='utils/log.txt'
)