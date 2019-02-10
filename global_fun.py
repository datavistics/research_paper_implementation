import logging
from functools import wraps
from sys import stdout
from timeit import default_timer as timer

from proj_paths import *


def module_logging(file, file_output: False = bool, level=logging.INFO):
    """
    Convenient logging handler
    :param file: Name of logger
    :type file: path-like
    :param file_output: Write to logs or not
    :type file_output: bool
    :param level: logging level
    :type level: int
    Examples
    --------
    >>> module_logger = module_logging(__file__, False, logging.INFO)
    >>> module_logger.info('This is an example')
    02-10 17:37:58 project.file_name INFO     This is an example
    """
    if isinstance(file, str):
        file = Path(file)

    logging.basicConfig(stream=stdout,
                        level=level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    module_logger = logging.getLogger(proj_dir.stem + '.' + str(file.stem))
    if file_output:
        fh = logging.FileHandler(file.with_suffix('.log'), mode='w')
        module_logger.addHandler(fh)
    return module_logger


# https://stackoverflow.com/a/26151604
def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def logspeed(f, module_logger):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = timer()
        result = f(*args, **kwargs)
        end = timer()
        module_logger.info(f'{f.__name__} - elapsed time: {end - start:.4f} seconds')
        return result

    return wrapper