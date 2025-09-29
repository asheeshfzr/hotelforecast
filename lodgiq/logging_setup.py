import logging
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    # Suppress noisy warnings in normal runs
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="np.find_common_type is deprecated", category=DeprecationWarning)
    return logger
