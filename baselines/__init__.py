"""Alias package for src.baselines to enable root-level 'import baselines'."""
from src.baselines import *
import src.baselines as _src_baselines
import sys

sys.modules["baselines"] = _src_baselines
