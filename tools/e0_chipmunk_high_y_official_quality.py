import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e0_chipmunk_high_y_metric_adapter import install
if __name__=='__main__':install('quality').main()
