# Factorial metrics: actual closed parent and review required.
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e0_chipmunk_p64_factorial_metric_adapter import install
if __name__=='__main__':install('raft').main()
