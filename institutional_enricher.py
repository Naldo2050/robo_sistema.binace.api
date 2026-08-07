import warnings
warnings.warn(
    "institutional_enricher na raiz está deprecated; importe de institutional.enricher",
    DeprecationWarning,
    stacklevel=2
)
from institutional.enricher import *
