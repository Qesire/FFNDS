"""P64-specific continuous RAFT entry point; actual closed inputs required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e0_chipmunk_p64_metric_adapter import install
if __name__=='__main__':install('raft').main()
