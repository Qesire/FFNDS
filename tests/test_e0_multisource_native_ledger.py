import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ffnds_observation_common import write_json
from run_e0_multisource_directional import reserve_attempt,close_attempt,PASS,ID


class FixedSourceLedger(unittest.TestCase):
    def test_corrupted_nonprefix_history_cannot_duplicate_a_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);plan={'id':ID,'pairs':[{'id':f'dc_source{i:02d}'} for i in [4,8,16,20]]}
            write_json(p/'STUDY.json',{'id':ID,'protocol_sha256':'fixed','attempts':[
                {'pair':'dc_source08','status':PASS}]})
            with self.assertRaises(RuntimeError):reserve_attempt(p,plan,'fixed',plan['pairs'][1])

    def test_running_failed_and_completed_attempt_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);plan={'id':ID,'pairs':[{'id':f'dc_source{i:02d}'} for i in [4,8,16,20]]}
            reserve_attempt(p,plan,'fixed',plan['pairs'][0])
            with self.assertRaises(RuntimeError):reserve_attempt(p,plan,'fixed',plan['pairs'][0])
            receipt=p/'RUN.json';write_json(receipt,{'status':'FAILED_PRESERVED_NO_RETRY'})
            close_attempt(p,plan['pairs'][0]['id'],receipt)
            with self.assertRaises(RuntimeError):reserve_attempt(p,plan,'fixed',plan['pairs'][1])

    def test_success_advances_exact_source_order_without_resetting_scope(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);plan={'id':ID,'pairs':[{'id':f'dc_source{i:02d}'} for i in [4,8,16,20]]}
            with self.assertRaises(RuntimeError):reserve_attempt(p,plan,'fixed',plan['pairs'][1])
            reserve_attempt(p,plan,'fixed',plan['pairs'][0])
            receipt=p/'RUN.json';write_json(receipt,{'status':PASS});close_attempt(p,plan['pairs'][0]['id'],receipt)
            with self.assertRaises(RuntimeError):reserve_attempt(p,plan,'other',plan['pairs'][1])
            reserve_attempt(p,plan,'fixed',plan['pairs'][1])


if __name__=='__main__':unittest.main()
