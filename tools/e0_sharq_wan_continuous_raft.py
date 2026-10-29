"""Actual original continuous RAFT, two exposed SharQ Wan source shards."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e0_sharq_wan_metric_adapter import install
if __name__=='__main__':install('raft').main()
