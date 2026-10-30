import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e0_chipmunk_input_video_quality_metric_adapter_v2 import install
if __name__=='__main__':install('raft').main()
