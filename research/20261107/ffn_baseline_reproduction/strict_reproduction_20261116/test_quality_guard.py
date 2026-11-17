"""Quality guard rejection before any model/Slurm access."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from quality_guard import main
class GuardTest(unittest.TestCase):
 def test_unapproved_scope_stops(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);scope=p/'scope.json';review=p/'REVIEW.json';active=p/'ACTIVE.json'
   review.write_text(json.dumps({'scope_sha256':'wrong','recheck_disposition':'REJECT','reviewer_agent_id':'reviewer'}));active.write_text(json.dumps({'key':{'scope_sha256':'wrong','review_sha256':'wrong','runnable':True}}))
   scope.write_text(json.dumps({'phase':'FFNDS_STAGE4_QUALITY','status':'REVIEW_CANDIDATE','review_path':str(review),'active_path':str(active),'active_key':'key','authors':['author']}))
   digest=hashlib.sha256(scope.read_bytes()).hexdigest()
   with self.assertRaises(RuntimeError):main(scope,digest)
if __name__=='__main__':unittest.main()
