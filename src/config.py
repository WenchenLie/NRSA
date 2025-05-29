import sys
from typing import Literal
from loguru import logger as LOGGER


LOGGER.remove()
LOGGER.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <red>|</red> <level>{level}</level> <red>|</red> <level>{message}</level>",
    level="DEBUG"
)

VERSION = '1.0.0'
AVAILABLE_SOLVERS = ['auto', 'Newmark-Newton', 'OPS']
SOLVER_TYPING = Literal['auto', 'Newmark-Newton', 'OPS']
AVAILABLE_ANA_TYPES = ['CDA', 'CSA']
ANA_TYPE_NAME = {'CDA': 'Constant ductility analysis', 'CSA': 'Constant strength analysis'}
