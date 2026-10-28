"""Isolated quality v3: preserve v2 scoring; interpret real dir_fd audit paths.

Only os.remove/os.rmdir audit observations with an existing directory FD are
normalized. The original OS call and every other audit event are untouched.
The original v2/v8 source files remain immutable. No numerical library is
imported by this wrapper or by byte preflight/inspection.
"""
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_COMPACT6016_CONSUMER00_12_QUALITY_DIRFD_V3_20261028"
V2_ID = "E0_COMPACT6016_CONSUMER00_12_QUALITY_ISOLATED_V2_20261028"
BASE_PATH = ROOT / "tools/e0_compact6016_consumer_quality.py"
BASE_SHA = "0e047e79cc77b91fa0bb45b88c065867f181315c92558ab56957c5f9920c187f"
V8_SHA = "5c4c0a3d3efc0c74d0acdac955bf844c739786f0833f2aed88fc2adfeb970257"
V2_PROTOCOL_SHA = "1a547c32982d9378a8f013a50d750f9d889df0f6d13880624baca9169c5fb352"
V2_RUN_SHA = "588daac050ad0977c7371359dad3c016e6daf032348a340731f386f3e6aca29d"
V2_EXIT_SHA = "0e8290ba464bc7e4003afb1919d83f5b9a1bfc8dc50bf9503b9f1af7d374d106"
FILELOCK_SHA = "379d37ee7580cf6005e7d2e5c8f32c3a0e08c062f5dc3eb1eb88614049f7bd04"
POLICY = {"id":"EXISTING_DIRECTORY_FD_REMOVE_RMDIR_ABSOLUTE_OBSERVER_V1",
          "events":["os.remove","os.rmdir"],"FD_resolution":"Linux /proc/self/fd + fstat/path directory identity",
          "require_FD_and_resolved_target_inside_scope":True,
          "original_audit_delegated":True,"original_OS_arguments_changed":False,
          "other_events_unchanged":True,"symlink_target_checks":"original resolved-target policy retained"}


def load_base():
    import hashlib
    if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA:
        raise ValueError("Immutable quality v2 source differs")
    spec=importlib.util.spec_from_file_location("_ffnds_quality_v2_for_v3",BASE_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


base=load_base()
base.ID=ID
base.REQUIRED_SOURCES=base.REQUIRED_SOURCES | {
    "tools/e0_compact6016_consumer_quality_v3.py","tests/test_e0_compact6016_consumer_quality_v3.py"}
_validate_plan=base.validate_plan
_verify_assets=base.verify_assets
_verify_environment=base.verify_environment
_cpu_report=base.cpu_report
_import_report=base.import_report
_require_import_receipt=base.require_import_receipt
_make_draft=base.make_draft
_run=base.run


def normalize_dir_fd_audit(original,scope,result,denied):
    """Correct observer coordinates; keep the original scope audit in charge.

    Audit arguments are (path, dir_fd). -1/None mean ordinary cwd resolution
    and are delegated unchanged. Existing FDs must refer to an in-scope live
    directory. Identity is checked against the resolved /proc link, including
    a second fstat; closed/reused/non-directory/outside descriptors fail shut.
    This application guard is not a kernel/native-I/O sandbox.
    """
    scope=Path(scope).resolve()
    result.setdefault("dir_fd_normalization",{"policy":POLICY,"normalized_events":0,"events":[]})
    record=result["dir_fd_normalization"]
    def fail(reason):
        denied("DIR_FD_AUDIT:"+reason)()
        raise RuntimeError("Directory-FD denial callable returned")
    def audit(event,values):
        if event not in POLICY["events"] or len(values)<2 or values[1] is None or values[1]==-1:
            return original(event,values)
        path,fd=values[0],values[1]
        if type(fd) is not int or fd<0 or not isinstance(path,(str,bytes,os.PathLike)):
            return fail("INVALID_ARGUMENTS")
        try:
            before=os.fstat(fd)
            if not stat.S_ISDIR(before.st_mode) or before.st_nlink==0:
                return fail("NOT_LIVE_DIRECTORY")
            link=Path("/proc/self/fd")/str(fd)
            raw=os.readlink(link)
            if not raw.startswith("/") or raw.endswith(" (deleted)"):
                return fail("UNRESOLVED_OR_DELETED_FD")
            directory=Path(raw).resolve(strict=True)
            actual=directory.stat();after=os.fstat(fd)
            identity=lambda value:(value.st_dev,value.st_ino)
            if identity(before)!=identity(actual) or identity(before)!=identity(after) or not directory.is_dir():
                return fail("FD_DIRECTORY_IDENTITY_CHANGED")
            if not directory.is_relative_to(scope):
                return fail("FD_OUTSIDE_SCOPE")
            requested=Path(os.fsdecode(path))
            target=(requested if requested.is_absolute() else directory/requested).resolve()
            if not target.is_relative_to(scope):
                return fail("TARGET_OUTSIDE_SCOPE")
        except OSError:
            return fail("INVALID_OR_UNRESOLVABLE_FD")
        normalized=(str(target),)+tuple(values[1:])
        # Record the actual bounded observation, then keep *all* original
        # checkpoint/network/filesystem checks; never mutate values or syscall.
        record["normalized_events"]+=1
        record["events"].append({"event":event,"dir_fd":fd,"directory":str(directory),
                                "original_path":os.fsdecode(path),"resolved_path":str(target)})
        return original(event,normalized)
    return audit


class GuardedEnvironment:
    def __init__(self,module):self.module=module
    def __getattr__(self,name):return getattr(self.module,name)
    def import_audit_guard(self,directory,result):
        denied,audit,font=self.module.import_audit_guard(directory,result)
        return denied,normalize_dir_fd_audit(audit,directory,result,denied),font


def validate_plan(plan,*,frozen=False):
    _validate_plan(plan,frozen=frozen)
    base.require(plan.get("directory_fd_audit_policy")==POLICY
        and plan.get("numeric_path")=="HIF4_FAKE_QDQ_DIAGNOSTIC"
        and plan.get("failed_import_predecessor_v2",{}).get("protocol",{}).get("sha256")==V2_PROTOCOL_SHA
        and plan["failed_import_predecessor_v2"].get("RUN",{}).get("sha256")==V2_RUN_SHA
        and plan["failed_import_predecessor_v2"].get("exit",{}).get("sha256")==V2_EXIT_SHA,
        "Fixed dir_fd v3 adapter/failure identity differs")
    probe=plan.get("directory_fd_filelock_probe_source",{})
    base.require(probe.get("root") in ("REPO","HF")
        and probe.get("sha256")==FILELOCK_SHA,"Actual filelock probe source differs")
    v8=[b for b in plan["repo_sources"] if b["path"]=="tools/e0_official_metric_environment_v8.py"]
    old=[b for b in plan["repo_sources"] if b["path"]=="tools/e0_compact6016_consumer_quality.py"]
    base.require(len(v8)==len(old)==1 and v8[0]["sha256"]==V8_SHA and old[0]["sha256"]==BASE_SHA,
                 "Immutable v2/v8 source bindings changed")


def verify_failed_import_v2(plan,hf):
    item=plan["failed_import_predecessor_v2"]
    protocol=base.read_bound(ROOT,item["protocol"])
    run=base.read_bound(hf,item["RUN"]);receipt=base.read_bound(hf,item["exit"])
    base.bound(hf,item["stderr"])
    base.require(protocol.get("id")==run.get("id")==receipt.get("id")==V2_ID
        and run.get("status")=="FAILED_IMPORTS_ONLY_PRESERVED_NO_RETRY"
        and run.get("protocol_sha256")==receipt.get("protocol_sha256")==V2_PROTOCOL_SHA
        and type(receipt.get("child_returncode")) is int and receipt["child_returncode"]==1
        and receipt.get("process_exited") is True and receipt.get("timeout") is False
        and receipt.get("automatic_retry") is False and receipt.get("run_sha256")==V2_RUN_SHA,
        "Actual failed v2 imports exit1 identity differs")
    base.require(run.get("predebited_official_module_imports")==1 and run.get("completed_official_module_imports")==0
        and run.get("functions")==[] and run.get("CUDA_initialized") is False
        and run.get("denied_attempts")==["FILESYSTEM_MUTATION_OUTSIDE_SCOPE"]
        and all(run.get(k)==0 for k in ("GPU_calls","checkpoint_loads","model_constructions",
            "parameter_creations","buffer_creations","model_forwards","official_scores")),
        "Failed v2 work/reason differs; no implicit retry authorization")
    frames=[f for row in run.get("denied_attempt_details",[]) for f in row.get("stack",[])]
    base.require(any(f.get("function")=="_probe_link_follow_symlinks" and f.get("file","").endswith("/filelock/_strict.py") for f in frames)
        and any(f.get("function")=="_rmtree_safe_fd" for f in frames),"Exact filelock/dir_fd failure trace absent")
    return item


def verify_assets(plan,hf,*,repo=ROOT):
    answer=_verify_assets(plan,hf,repo=repo)
    verify_failed_import_v2(plan,hf)
    verify_probe_source(plan,hf)
    return answer


def verify_environment(plan,hf):
    helper,v8=_verify_environment(plan,hf)
    base.require(base.sha(Path(helper.__file__))==V8_SHA,"Original v8 source changed")
    verify_probe_source(plan,hf)
    verify_failed_import_v2(plan,hf)
    return GuardedEnvironment(helper),v8


def verify_probe_source(plan,hf):
    item=plan["directory_fd_filelock_probe_source"]
    actual=base.bound(ROOT if item["root"]=="REPO" else hf,item)
    expected=(Path(hf)/plan["environment"]["site"]/"filelock/_strict.py").resolve()
    base.require(actual==expected,"Selected original filelock probe symlink origin differs")
    return actual


def proof_extension(plan):
    return {"directory_fd_audit_policy_sha256":base.state_sha(plan["directory_fd_audit_policy"]),
            "failed_import_v2_RUN_sha256":plan["failed_import_predecessor_v2"]["RUN"]["sha256"],
            "filelock_probe_source_sha256":FILELOCK_SHA,
            "numeric_path":"HIF4_FAKE_QDQ_DIAGNOSTIC"}


def cpu_report(plan,digest):return {**_cpu_report(plan,digest),**proof_extension(plan)}
def import_report(plan,digest):return {**_import_report(plan,digest),**proof_extension(plan)}


def require_import_receipt(scope,plan,digest,hf):
    proof=_require_import_receipt(scope,plan,digest,hf)
    report=base.read_bound(hf,proof["RUN"])
    normalized=report.get("dir_fd_normalization",{})
    base.require(normalized.get("policy")==POLICY and type(normalized.get("normalized_events")) is int
        and normalized["normalized_events"]==len(normalized.get("events",[]))>0,
        "Actual successful dir_fd cleanup observations absent")
    return proof


def run(args):
    # The native evaluator's later filesystem audit uses the same path
    # convention. Keep its original process/network allowlist intact too.
    import importlib
    common=importlib.import_module("e0_ub1_consumer_official_quality")
    factory=common.audit_scope;observations={"denied_attempts":[]}
    def denied(reason):
        def fail():observations["denied_attempts"].append(reason);raise RuntimeError(reason)
        return fail
    def wrapped_factory(scope,*arguments):
        return normalize_dir_fd_audit(factory(scope,*arguments),scope,observations,denied)
    plan=base.read(args.protocol)
    out=args.hf_root.resolve()/plan["output_HF_relative"]
    base.require(not out.exists(),"Fresh v3 output required; no failed-output overwrite")
    common.audit_scope=wrapped_factory
    try:
        result=_run(args)
        if observations["denied_attempts"]:
            result["status"]="FAILED_V3_DIR_FD_AUDIT_PRESERVED_NO_RETRY"
            base.write(out/"RUN.json",result)
            raise RuntimeError("Native directory-FD audit violation retained")
        return result
    finally:
        common.audit_scope=factory
        path=out/"RUN.json"
        if path.is_file():
            report=base.read(path)
            base.require(report.get("id")==ID and report.get("protocol_sha256")==base.sha(args.protocol),
                         "Refuse to modify a different attempt result")
            report.update(proof_extension(plan),native_dir_fd_audit=observations)
            base.write(path,report)


def make_draft(hf,*,repo=ROOT):
    plan=_make_draft(hf,repo=repo)
    previous=Path(hf)/"ffnds-derived/e0_compact6016_quality_isolated_cpu_20261028/v2/imports/worker"
    probe=(Path(hf)/plan["environment"]["site"]/"filelock/_strict.py").resolve()
    probe_root=Path(repo).resolve() if probe.is_relative_to(Path(repo).resolve()) else Path(hf).resolve()
    probe_binding={"root":"REPO" if probe_root==Path(repo).resolve() else "HF",**base.binding(probe,probe_root)}
    plan.update(id=ID,status="DRAFT_DIR_FD_OBSERVER_FIXED_SIX_SAME_ENVIRONMENT_QUALITY",
        numeric_path="HIF4_FAKE_QDQ_DIAGNOSTIC",directory_fd_audit_policy=POLICY,
        directory_fd_filelock_probe_source=probe_binding,
        output_HF_relative="ffnds-derived/e0_compact6016_consumer00_12_quality_dirfd_v3_20261028/attempt1/results",
        failed_import_predecessor_v2={"protocol":base.binding(Path(repo)/"config/e0_compact6016_consumer_quality_20261028_v2.json",repo),
            "RUN":base.binding(previous/"imports/RUN.json",hf),"exit":base.binding(previous/"EXIT_RECEIPT.json",hf),
            "stderr":base.binding(previous/"child.stderr.log",hf),
            "identity":"V2_IMPORTS_EXIT1_PREDEBIT1_COMPLETED0_ZERO_NN_SCORES_CUDA_FILELOCK_DIR_FD_CLEANUP"})
    plan["pending_freeze"]="Independent review, exact v3 source/protocol registration and new actual exited byte-only plus eight-module guarded imports-only CPU receipts. Failed v1/v2 records remain unchanged. No automatic retry."
    return plan


base.validate_plan=validate_plan
base.verify_assets=verify_assets
base.verify_environment=verify_environment
base.cpu_report=cpu_report
base.import_report=import_report
base.require_import_receipt=require_import_receipt
base.make_draft=make_draft
base.run=run

if __name__=="__main__":base.main()
