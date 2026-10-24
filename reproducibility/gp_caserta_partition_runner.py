"""Condition-sharded Caserta GP × UWSAA evaluation wrapper."""
from __future__ import annotations
import argparse, concurrent.futures, ctypes, hashlib, json, os, queue, shutil, subprocess, sys, threading, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from environment.online_crp import load_episode_manifest
from reproducibility.protocol import CONDITION_SETS
from reproducibility.runner import _gp_model_record
from uwsaa.entrypoint import load_gp_rule

ARMS = ("NATIVE20", "UWSAA")
PARTITIONS = {"local": tuple(range(1, 15)), "cloud": tuple(range(15, 42)), "all": tuple(range(1, 42))}
PROBE_CONDITIONS = ("CLEAN", "subset_shuffle|p=0.3|repeat=0", "time_window_shuffle|B=5|repeat=0",
    "delayed_positions|p=0.3|repeat=0", "overlay_mix|p=0.3|B=5|repeat=0",
    "regime_mix|p=0.3|B=5|profile=C_EQ|repeat=0")
PROBE_FIELDS = ("episode_id","condition_id","arm","layout_fingerprint","parameter_group","num_containers",
    "relocations","relocations_per_container","decisions","num_retrieved","provider_reveal_count",
    "forecast_table_sha256","gp_model")
DEFAULT_DATA = ROOT / "data" / "caserta_table2_840.json"
DEFAULT_REFERENCE = ROOT / "reproducibility" / "references" / "gp_caserta_crosshost_probe_reference.json"

def utc_now(): return datetime.now(timezone.utc)

def sha_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()

def atomic_json(path: Path, value: Mapping[str,Any]):
    tmp=path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    if json.loads(path.read_text(encoding="utf-8"))!=value: raise RuntimeError(f"atomic read-back failed: {path.name}")

def runtime_inventory():
    paths=set()
    for base in ("baseline","environment","uwsaa","third_party/crp_online/src"):
        paths.update(p for p in (ROOT/base).rglob("*.py") if p.is_file())
    paths.update(ROOT/p for p in ("reproducibility/runner.py","reproducibility/protocol.py",
        "configs/uwsaa.json","configs/gp_rule.json","data/caserta_table2_840.json"))
    files,digest={},hashlib.sha256()
    for p in sorted(paths,key=lambda x:x.relative_to(ROOT).as_posix()):
        rel=p.relative_to(ROOT).as_posix(); value=sha_file(p); files[rel]=value
        digest.update(rel.encode()); digest.update(b"\0"); digest.update(value.encode()); digest.update(b"\n")
    return {"files":files,"closure_sha256":digest.hexdigest()}

def assert_runtime_current(expected, data_sha):
    if runtime_inventory()!=expected: raise RuntimeError("runtime inventory drift before dispatch")
    cfg=json.loads((ROOT/"configs/uwsaa.json").read_text(encoding="utf-8"))
    if cfg.get("dataset_sha256")!=data_sha: raise RuntimeError("uwsaa config dataset SHA differs from selected Caserta file")

def cpu_quota():
    limits=[max(1,os.cpu_count() or 1)]
    if hasattr(os,"sched_getaffinity"):
        try: limits.append(max(1,len(os.sched_getaffinity(0))))
        except OSError: pass
    if os.environ.get("CODEX_CPU_QUOTA_CORES"):
        limits.append(max(1,int(float(os.environ["CODEX_CPU_QUOTA_CORES"]))))
    for p in (Path("/sys/fs/cgroup/cpu.max"),Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")):
        try:
            raw=p.read_text(encoding="ascii").strip()
            if p.name=="cpu.max":
                q,period=raw.split()[:2]
                if q!="max": limits.append(max(1,int(int(q)/int(period))))
            else:
                q=int(raw); period=int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
                if q>0: limits.append(max(1,int(q/period)))
        except (OSError,ValueError,ZeroDivisionError): pass
    return min(limits)

def keep_awake():
    if os.name!="nt": return None
    setter=ctypes.windll.kernel32.SetThreadExecutionState
    setter.argtypes=[ctypes.c_uint]; setter.restype=ctypes.c_uint
    if not setter(0x80000001): raise ctypes.WinError()
    return setter

def restore_awake(setter):
    if setter is not None: setter(0x80000000)

def actual_episodes(data):
    from reproducibility.gp_arm_paired_analysis import actual_episode_index
    return tuple(load_episode_manifest(data)),actual_episode_index(data)

def first_per_geometry(episodes):
    result={}
    for e in episodes: result.setdefault(str(e.parameter_group),str(e.episode_id))
    if len(result)!=21: raise ValueError("probe requires the 21 registered geometries")
    return tuple(result.values())

def validate_shard(endpoint:Path,data:Path,data_sha:str,gp:Mapping[str,Any],
                    condition_ids:tuple[str,...],episode_ids:tuple[str,...]|None=None):
    from reproducibility import gp_arm_paired_analysis as analysis
    mp=Path(str(endpoint)+".manifest.json")
    if not endpoint.is_file() or not mp.is_file(): raise ValueError(f"missing endpoint/manifest: {endpoint.name}")
    m=json.loads(mp.read_text(encoding="utf-8"))
    registry={str(c["condition_id"]):dict(c) for c in CONDITION_SETS["main"]}
    expected_conditions=[registry[c] for c in condition_ids]
    if (m.get("schema")!="uwsaa_trajectory_run_manifest_v1" or m.get("status")!="complete"
        or m.get("smoke_only") is not False or m.get("suite")!="main" or m.get("dataset_sha256")!=data_sha
        or m.get("gp_model")!=gp or m.get("solvers")!=["GP"] or m.get("arms")!=list(ARMS)
        or m.get("conditions")!=expected_conditions or m.get("endpoint_file")!=endpoint.name
        or int(m.get("failed_rows",-1))!=0):
        raise ValueError(f"runner manifest contract failed: {endpoint.name}")
    actual=analysis.actual_episode_index(data)
    selected=tuple(episode_ids) if episode_ids is not None else tuple(actual)
    if len(selected)!=len(set(selected)) or any(x not in actual for x in selected): raise ValueError("invalid episode subset")
    expected_eps={x:actual[x] for x in selected}
    expected_manifest=[{"episode_id":x,"layout_fingerprint":actual[x]["layout_fingerprint"],
        "parameter_group":actual[x]["geometry"],"num_containers":actual[x]["num_containers"]} for x in selected]
    if m.get("episodes")!=expected_manifest: raise ValueError(f"manifest episode fingerprints differ: {endpoint.name}")
    rows=[json.loads(line) for line in endpoint.read_text(encoding="utf-8").splitlines()]
    expected_n=len(selected)*len(condition_ids)*len(ARMS)
    if (len(rows)!=expected_n or int(m.get("expected_rows",-1))!=expected_n
        or int(m.get("actual_rows",-1))!=expected_n): raise ValueError(f"row count differs: {endpoint.name}")
    for r in rows:
        cid,eid=str(r.get("condition_id","")),str(r.get("episode_id",""))
        if cid not in condition_ids or eid not in expected_eps: raise ValueError(f"unexpected endpoint key {cid}/{eid}")
        if r.get("schema")!="uwsaa_trajectory_endpoint_v1" or r.get("solver")!="GP" or r.get("arm") not in ARMS:
            raise ValueError(f"endpoint schema/solver/arm mismatch: {cid}/{eid}")
        analysis.validate_endpoint_row(r,expected_eps[eid],registry[cid],gp)
    analysis.validate_pairs(rows,set(condition_ids),expected_eps,gp)
    return {"endpoint_sha256":sha_file(endpoint),"manifest_sha256":sha_file(mp),
        "rows":expected_n,"endpoint_file":endpoint.name,"manifest_file":mp.name},rows

def load_reference(path:Path,data_sha:str,gp:Mapping[str,Any],episodes):
    ref=json.loads(path.read_text(encoding="utf-8"))
    if ref.get("schema")!="gp_caserta_crosshost_probe_reference_v1": raise ValueError("unsupported probe reference")
    if ref.get("dataset_sha256")!=data_sha or ref.get("gp_model")!=gp: raise ValueError("probe data/model identity drift")
    if tuple(ref.get("condition_ids",()))!=PROBE_CONDITIONS: raise ValueError("probe conditions differ")
    episode_ids=first_per_geometry(episodes)
    if tuple(ref.get("episode_ids",()))!=episode_ids: raise ValueError("probe episode selection differs")
    expected={(e,c,a) for e in episode_ids for c in PROBE_CONDITIONS for a in ARMS}; keyed={}
    for r in ref.get("records",[]):
        if not isinstance(r,dict) or set(r)!=set(PROBE_FIELDS): raise ValueError("malformed probe row")
        key=(str(r["episode_id"]),str(r["condition_id"]),str(r["arm"]))
        if key not in expected or key in keyed: raise ValueError("duplicate/unexpected probe key")
        keyed[key]=r
    if set(keyed)!=expected: raise ValueError(f"probe reference has {len(keyed)} records, expected 252")
    return ref

def run_probe(args,data_sha,gp,episodes,workers,inventory):
    import platform
    import numpy as np
    ref=load_reference(args.probe_reference.resolve(),data_sha,gp,episodes)
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError("probe output directory must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    ids=tuple(ref["episode_ids"]); conditions=tuple(ref["condition_ids"])
    global_index={str(c["condition_id"]):i for i,c in enumerate(CONDITION_SETS["main"],1)}
    deadline=time.monotonic()+args.wall_seconds; started=time.perf_counter(); start_utc=utc_now()
    active=set(); active_lock=threading.RLock(); peak=[0]; stop=threading.Event()
    def run_condition(cid):
        index=global_index[cid]
        endpoint=out/f"probe-condition-{index:02d}.jsonl"
        log=endpoint.with_suffix(endpoint.suffix+".runner.log")
        command=[sys.executable,"-X","utf8","-B",str(ROOT/"reproducibility"/"runner.py"),
            "--run","--suite","main","--condition",cid,"--solver","GP","--arm","NATIVE20","--arm","UWSAA",
            "--data",str(args.data.resolve()),"--output",str(endpoint)]
        for eid in ids: command.extend(("--episode-id",eid))
        assert_runtime_current(inventory,data_sha)
        proc=None; stream=None; began=time.perf_counter(); began_utc=utc_now()
        try:
            with active_lock:
                if stop.is_set(): raise RuntimeError("probe cancelled after another shard failed")
                stream=log.open("xb")
                proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
                active.add(proc); peak[0]=max(peak[0],len(active))
            while proc.poll() is None:
                remain=deadline-time.monotonic()
                if stop.is_set() or remain<=0:
                    proc.kill()
                    if remain<=0: raise TimeoutError("probe wall cap reached")
                    raise RuntimeError("probe shard cancelled")
                time.sleep(min(.2,remain))
            code=proc.wait()
            if code: raise RuntimeError(f"probe shard {cid} exited {code}; see {log.name}")
            detail,rows=validate_shard(endpoint,args.data.resolve(),data_sha,gp,(cid,),ids)
            return {"global_condition_index":index,"condition_id":cid,"endpoint_file":endpoint.name,
                "manifest_file":Path(str(endpoint)+".manifest.json").name,"endpoint_sha256":detail["endpoint_sha256"],
                "manifest_sha256":detail["manifest_sha256"],"rows":detail["rows"],"child_pid":proc.pid,
                "started_utc":began_utc.isoformat(timespec="seconds"),"elapsed_seconds":time.perf_counter()-began},rows
        finally:
            if proc is not None:
                with active_lock: active.discard(proc)
            if stream is not None: stream.close()
    outcomes={}; failures=[]
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures={pool.submit(run_condition,cid):cid for cid in conditions}
    try:
        for future in concurrent.futures.as_completed(futures):
            cid=futures[future]
            try: outcomes[cid]=future.result()
            except BaseException as exc:
                failures.append({"condition_id":cid,"error_type":type(exc).__name__,"error":str(exc)})
                stop.set()
                with active_lock: running=list(active)
                for proc in running:
                    if proc.poll() is None:
                        try:proc.kill()
                        except OSError:pass
                for pending in futures: pending.cancel()
                break
        pool.shutdown(wait=True,cancel_futures=True)
    except BaseException:
        stop.set()
        with active_lock: running=list(active)
        for proc in running:
            if proc.poll() is None:
                try:proc.kill()
                except OSError:pass
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    elapsed=time.perf_counter()-started
    rows=[]; shards=[]
    for cid in conditions:
        if cid not in outcomes: continue
        shard,part=outcomes[cid]; shards.append(shard); rows.extend(part)
    reference={(r["episode_id"],r["condition_id"],r["arm"]):r for r in ref["records"]}
    actual={(str(r["episode_id"]),str(r["condition_id"]),str(r["arm"])):r for r in rows}
    mismatches=[]
    if set(actual)!=set(reference): mismatches.append({"type":"key_set","expected":len(reference),"actual":len(actual)})
    for key in set(actual)&set(reference):
        record={field:actual[key].get(field) for field in PROBE_FIELDS}
        if record!=reference[key]:
            mismatches.append({"key":list(key),"fields":[field for field in PROBE_FIELDS if record[field]!=reference[key][field]]})
            if len(mismatches)>=20: break
    if not failures and not mismatches and len(rows)==252:
        merged=out/"probe.jsonl"
        with merged.open("x",encoding="utf-8",newline="\n") as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
        episode_meta={}
        for row in rows:
            episode_meta.setdefault(row["episode_id"],{"episode_id":row["episode_id"],
                "layout_fingerprint":row["layout_fingerprint"],"parameter_group":row["parameter_group"],
                "num_containers":row["num_containers"]})
        probe_manifest={"schema":"gp_caserta_crosshost_probe_manifest_v1","status":"complete",
            "suite":"main","solvers":["GP"],"arms":list(ARMS),"conditions":[dict(c) for c in CONDITION_SETS["main"] if c["condition_id"] in conditions],
            "episodes":list(episode_meta.values()),"dataset_sha256":data_sha,"gp_model":gp,
            "expected_rows":252,"actual_rows":len(rows),"failed_rows":0,"endpoint_file":merged.name,
            "probe_shards":shards}
        atomic_json(Path(str(merged)+".manifest.json"),probe_manifest)
    else:
        merged=out/"probe.jsonl"
    affinity=None
    if hasattr(os,"sched_getaffinity"):
        try: affinity=len(os.sched_getaffinity(0))
        except OSError: pass
    if affinity is None: affinity=os.cpu_count()
    thread_names=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")
    receipt={"schema":"gp_caserta_crosshost_probe_receipt_v1","status":"pass" if not failures and not mismatches and len(rows)==252 else "failed",
        "dataset_sha256":data_sha,"gp_model":gp,"reference_sha256":sha_file(args.probe_reference.resolve()),
        "runtime_inventory":inventory,"partition_runner_sha256":sha_file(Path(__file__)),
        "validation_script_sha256":sha_file(ROOT/"reproducibility/gp_arm_paired_analysis.py"),
        "runner_sha256":sha_file(ROOT/"reproducibility/runner.py"),
        "condition_ids":list(conditions),"episode_ids":list(ids),"matched_records":len(rows),
        "requested_workers":args.workers if args.workers is not None else workers,"worker_count":workers,
        "peak_simultaneous_children":peak[0],"wall_seconds":elapsed,"started_utc":start_utc.isoformat(timespec="seconds"),
        "finished_utc":utc_now().isoformat(timespec="seconds"),"cpu_quota_cores":cpu_quota(),
        "process_affinity_cores":affinity,"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),
        "thread_caps":{name:os.environ.get(name) for name in thread_names},"shards":shards,
        "failures":failures,"mismatches":mismatches,"merged_endpoint_file":merged.name if merged.is_file() else None}
    atomic_json(out/"probe_receipt.json",receipt)
    if receipt["status"]!="pass": raise RuntimeError("cross-host probe failed; do not launch cloud formal partition")
    return receipt

def parse_utc(value:str):
    dt=datetime.fromisoformat(value.replace("Z","+00:00"))
    if dt.tzinfo is None: raise ValueError("deadline needs UTC offset")
    return dt.astimezone(timezone.utc)

def fresh_state(out,partition,indexes,workers,data_sha,gp,inventory,deadline,probe_sha):
    return {"schema":"gp_caserta_partition_run_v1","status":"running","partition":partition,
        "condition_indexes":list(indexes),"conditions":[dict(CONDITION_SETS["main"][i-1]) for i in indexes],
        "suite":"main","solver":"GP","arms":list(ARMS),"dataset_sha256":data_sha,"expected_episodes":840,
        "expected_endpoint_rows":1680*len(indexes),"completed_conditions":[],"completed_endpoint_rows":0,
        "shards":[],"attempts":[],"failures":[],"worker_count":workers,"runtime_inventory":inventory,
        "partition_runner_sha256":sha_file(Path(__file__)),"validation_script_sha256":sha_file(ROOT/"reproducibility/gp_arm_paired_analysis.py"),"gp_model":gp,"probe_receipt_sha256":probe_sha,
        "started_utc":utc_now().isoformat(timespec="seconds"),"deadline_utc":deadline.isoformat(timespec="seconds"),
        "deadline_epoch_seconds":deadline.timestamp(),"output_root_name":out.name,"python":sys.version,
        "heartbeat_utc":utc_now().isoformat(timespec="seconds"),"active_children":[]}

def validate_saved(out,state,data,data_sha,gp):
    seen=set(); ids=set(); indexes=tuple(int(i) for i in state["condition_indexes"])
    for r in state.get("shards",[]):
        i,cid=int(r["global_condition_index"]),str(r["condition_id"])
        if i not in indexes or CONDITION_SETS["main"][i-1]["condition_id"]!=cid or i in seen: raise ValueError("invalid shard receipt")
        ep=out/r["endpoint_file"]; mp=Path(str(ep)+".manifest.json")
        if sha_file(ep)!=r["endpoint_sha256"] or sha_file(mp)!=r["manifest_sha256"]: raise ValueError(f"shard SHA drift: {i}")
        detail,_=validate_shard(ep,data,data_sha,gp,(cid,))
        if detail["endpoint_sha256"]!=r["endpoint_sha256"] or detail["manifest_sha256"]!=r["manifest_sha256"]:
            raise ValueError(f"shard changed during validation: {i}")
        seen.add(i); ids.add(cid)
    if set(state.get("completed_conditions",()))!=ids or int(state.get("completed_endpoint_rows",-1))!=1680*len(seen):
        raise ValueError("completed row/receipt mismatch")
    return seen

def assert_resume_artifacts(out,state,indexes):
    if state.get("active_children") or state.get("current_child_pid") is not None:
        raise ValueError("resume refused: stale active-child state requires process/output review")
    allowed={}
    for receipt in state.get("shards",[]):
        index=int(receipt["global_condition_index"])
        allowed[index]={receipt["endpoint_file"],receipt["manifest_file"],receipt["endpoint_file"]+".runner.log"}
    unexpected=[]
    for index in range(1,42):
        candidates=set(out.glob(f"condition-{index:02d}*.jsonl"))
        candidates.update(out.glob(f"condition-{index:02d}*.jsonl.*"))
        for path in candidates:
            if index not in indexes or path.name not in allowed.get(index,set()):
                unexpected.append(path.name)
    if unexpected:
        raise ValueError("resume refused: unreceipted/incomplete shard outputs require review: "+", ".join(sorted(unexpected)[:12]))


def import_shards(src,out,state,data,data_sha,gp,inventory):
    source=json.loads((src/"run_manifest.json").read_text(encoding="utf-8"))
    if source.get("schema") not in ("gp_caserta_condition_shard_run_v1","gp_caserta_partition_run_v1"):
        raise ValueError("import source has unsupported manifest")
    if source.get("dataset_sha256")!=data_sha or source.get("gp_model")!=gp or source.get("runtime_inventory")!=inventory:
        raise ValueError("import dataset/model/runtime identity differs")
    source_shards=source.get("shards",[])
    receipt_ids=[str(r.get("condition_id","")) for r in source_shards]
    if len(receipt_ids)!=len(set(receipt_ids)) or set(source.get("completed_conditions",()))!=set(receipt_ids):
        raise ValueError("import completed-condition list differs from shard receipts")
    if int(source.get("completed_endpoint_rows",-1))!=1680*len(source_shards):
        raise ValueError("import completed-row count differs from shard receipts")
    source_shards=source.get("shards",[])
    receipt_ids=[str(r.get("condition_id","")) for r in source_shards]
    if len(receipt_ids)!=len(set(receipt_ids)) or set(source.get("completed_conditions",()))!=set(receipt_ids):
        raise ValueError("import completed-condition list differs from shard receipts")
    if int(source.get("completed_endpoint_rows",-1))!=1680*len(source_shards):
        raise ValueError("import completed-row count differs from shard receipts")
    receipts={}
    if source["schema"]=="gp_caserta_condition_shard_run_v1":
        if source.get("expected_conditions")!=[c["condition_id"] for c in CONDITION_SETS["main"]]:
            raise ValueError("source condition registry differs")
        for r in source.get("shards",[]):
            found=[i for i,c in enumerate(CONDITION_SETS["main"],1) if c["condition_id"]==r.get("condition_id")]
            if len(found)==1: receipts[found[0]]=r
    else: receipts={int(r["global_condition_index"]):r for r in source.get("shards",[])}
    imported=[]
    for i in sorted(set(state["condition_indexes"])&set(receipts)):
        rec=receipts[i]; cid=CONDITION_SETS["main"][i-1]["condition_id"]
        src_ep,src_mp=src/rec["endpoint_file"],src/rec["manifest_file"]
        if (not src_ep.is_file() or not src_mp.is_file() or sha_file(src_ep)!=rec.get("endpoint_sha256")
            or sha_file(src_mp)!=rec.get("manifest_sha256")): raise ValueError(f"source shard receipt failed: {i}")
        validate_shard(src_ep,data,data_sha,gp,(cid,))
        dst=out/src_ep.name; dst_mp=out/src_mp.name
        if dst.exists() or dst_mp.exists(): raise FileExistsError(f"refusing overwrite: {dst.name}")
        shutil.copyfile(src_ep,dst); shutil.copyfile(src_mp,dst_mp)
        if sha_file(dst)!=rec["endpoint_sha256"] or sha_file(dst_mp)!=rec["manifest_sha256"]: raise IOError("import copy hash mismatch")
        detail,_=validate_shard(dst,data,data_sha,gp,(cid,))
        imported.append({"global_condition_index":i,"condition_id":cid,"endpoint_file":dst.name,
            "manifest_file":dst_mp.name,"endpoint_sha256":detail["endpoint_sha256"],
            "manifest_sha256":detail["manifest_sha256"],"rows":detail["rows"],"status":"complete",
            "imported_from_run":src.name})
    state["shards"].extend(imported); state["shards"].sort(key=lambda r:r["global_condition_index"])
    state["completed_conditions"]=[r["condition_id"] for r in state["shards"]]
    state["completed_endpoint_rows"]=sum(r["rows"] for r in state["shards"])

def run_partition(args,indexes,workers,data,data_sha,gp,inventory,probe_sha):
    out=args.output.resolve(); run_path=out/"run_manifest.json"
    expected={"schema":"gp_caserta_partition_run_v1","partition":args.partition,"condition_indexes":list(indexes),
        "conditions":[dict(CONDITION_SETS["main"][i-1]) for i in indexes],"suite":"main","solver":"GP","arms":list(ARMS),
        "dataset_sha256":data_sha,"expected_episodes":840,"expected_endpoint_rows":1680*len(indexes),
        "runtime_inventory":inventory,"partition_runner_sha256":sha_file(Path(__file__)),"validation_script_sha256":sha_file(ROOT/"reproducibility/gp_arm_paired_analysis.py"),"gp_model":gp,"probe_receipt_sha256":probe_sha}
    if args.resume:
        if not run_path.is_file(): raise FileNotFoundError("--resume requires run_manifest.json")
        state=json.loads(run_path.read_text(encoding="utf-8"))
        for k,v in expected.items():
            if state.get(k)!=v: raise ValueError(f"resume refused: {k} drift")
        if state.get("status")=="complete": raise RuntimeError("partition already complete")
        done=validate_saved(out,state,data,data_sha,gp)
        if state.get("failures"): raise ValueError("resume refused: failed attempts require review; start a fresh output after preserving evidence")
        assert_resume_artifacts(out,state,indexes)
        deadline=parse_utc(state["deadline_utc"])
        if args.deadline_utc: deadline=min(deadline,parse_utc(args.deadline_utc))
        deadline=min(deadline,utc_now()+timedelta(seconds=args.wall_seconds))
        state["deadline_utc"],state["deadline_epoch_seconds"]=deadline.isoformat(timespec="seconds"),deadline.timestamp()
        state["worker_count"]=workers
    else:
        if out.exists() and any(out.iterdir()): raise FileExistsError("output nonempty; use --resume or a new directory")
        out.mkdir(parents=True,exist_ok=True); now=utc_now()
        deadline=parse_utc(args.deadline_utc) if args.deadline_utc else now+timedelta(seconds=args.wall_seconds)
        deadline=min(deadline,now+timedelta(seconds=args.wall_seconds))
        if deadline<=now: raise ValueError("absolute deadline has already expired")
        state=fresh_state(out,args.partition,indexes,workers,data_sha,gp,inventory,deadline,probe_sha); done=set()
        atomic_json(run_path,state)
        if args.import_shard_dir:
            import_shards(args.import_shard_dir.resolve(),out,state,data,data_sha,gp,inventory)
            done={int(r["global_condition_index"]) for r in state["shards"]}; atomic_json(run_path,state)
    if {int(r["global_condition_index"]) for r in state["shards"]}!=done: raise ValueError("receipt set mismatch")
    attempts={}
    for r in state.get("attempts",[]): attempts[int(r["global_condition_index"])]=attempts.get(int(r["global_condition_index"]),0)+1
    files={}
    for i in indexes:
        if i in done: continue
        n=attempts.get(i,0)+1; ep=out/(f"condition-{i:02d}.jsonl" if n==1 else f"condition-{i:02d}.attempt-{n:02d}.jsonl")
        while ep.exists() or Path(str(ep)+".manifest.json").exists():
            n+=1; ep=out/f"condition-{i:02d}.attempt-{n:02d}.jsonl"
        files[i]=ep
    todo=queue.Queue()
    for i in indexes:
        if i not in done: todo.put(i)
    stop=threading.Event(); slock=threading.RLock(); plock=threading.RLock(); active=set()
    def save():
        with slock: state["heartbeat_utc"]=utc_now().isoformat(timespec="seconds"); atomic_json(run_path,state)
    def terminate():
        stop.set()
        with plock: ps=list(active)
        for p in ps:
            if p.poll() is None:
                try:p.kill()
                except OSError:pass
        for p in ps:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
    def worker():
        while not stop.is_set():
            try:i=todo.get_nowait()
            except queue.Empty:return
            cond=CONDITION_SETS["main"][i-1]; ep=files[i]; log=ep.with_suffix(ep.suffix+".runner.log")
            proc=stream=None; started=time.perf_counter()
            try:
                assert_runtime_current(inventory,data_sha)
                with plock:
                    if stop.is_set(): return
                    stream=log.open("xb")
                    cmd=[sys.executable,"-X","utf8","-B",str(ROOT/"reproducibility"/"runner.py"),"--run","--suite","main",
                         "--condition",cond["condition_id"],"--solver","GP","--arm","NATIVE20","--arm","UWSAA",
                         "--data",str(data),"--output",str(ep)]
                    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
                    active.add(proc)
                with slock:
                    state["current_condition"]=cond["condition_id"]; state["current_child_pid"]=proc.pid
                    state["active_children"].append({"index":i,"pid":proc.pid,"condition_id":cond["condition_id"],
                        "started_utc":utc_now().isoformat(timespec="seconds")}); save()
                while proc.poll() is None:
                    if stop.is_set(): proc.kill(); break
                    remain=state["deadline_epoch_seconds"]-time.time()
                    if remain<=0: proc.kill(); raise TimeoutError("persistent deadline reached")
                    time.sleep(min(.25,remain))
                code=proc.wait()
                if code!=0: raise RuntimeError(f"runner exit {code}; see {log.name}")
                cid=cond["condition_id"]; detail,_=validate_shard(ep,data,data_sha,gp,(cid,))
                rec={"global_condition_index":i,"condition_id":cid,"endpoint_file":ep.name,
                    "manifest_file":Path(str(ep)+".manifest.json").name,"endpoint_sha256":detail["endpoint_sha256"],
                    "manifest_sha256":detail["manifest_sha256"],"rows":detail["rows"],"status":"complete",
                    "elapsed_seconds":time.perf_counter()-started,"child_pid":proc.pid}
                with slock:
                    state["shards"].append(rec); state["shards"].sort(key=lambda r:r["global_condition_index"])
                    state["completed_conditions"]=[r["condition_id"] for r in state["shards"]]
                    state["completed_endpoint_rows"]=sum(r["rows"] for r in state["shards"])
                    state["active_children"]=[x for x in state["active_children"] if x["pid"]!=proc.pid]
                    state["current_condition"]=None; state["current_child_pid"]=None; save()
                print(json.dumps({"event":"CONDITION_COMPLETE","index":i,"condition_id":cid,"rows":detail["rows"],
                    "elapsed_seconds":rec["elapsed_seconds"],"child_pid":proc.pid}),flush=True)
            except BaseException as exc:
                failure={"global_condition_index":i,"condition_id":cond["condition_id"],"endpoint_file":ep.name,
                    "log_file":log.name,"error_type":type(exc).__name__,"error":str(exc),
                    "recorded_utc":utc_now().isoformat(timespec="seconds")}
                with slock:
                    state["attempts"].append({**failure,"status":"failed"}); state["failures"].append(failure)
                    state["status"]="failed"; state["current_condition"]=cond["condition_id"]
                    state["current_child_pid"]=None if proc is None else proc.pid; save()
                terminate(); raise
            finally:
                if proc is not None:
                    with plock:active.discard(proc)
                if stream is not None:stream.close()
                todo.task_done()
    state["worker_count"]=workers; state["status"]="running"; save()
    print(json.dumps({"event":"PARTITION_RUNNING","partition":args.partition,"workers":workers,
        "conditions":len(indexes),"completed_before_start":len(done),"pending":todo.qsize(),
        "run_manifest":str(run_path),"deadline_utc":state["deadline_utc"],
        "dataset_sha256":data_sha,"runtime_closure_sha256":inventory["closure_sha256"],
        "gp_formula_sha256":gp["formula_sha256"]}),flush=True)
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers); futures=[pool.submit(worker) for _ in range(workers)]
    try:
        for f in concurrent.futures.as_completed(futures):
            if f.exception() is not None:
                terminate()
                for other in futures:other.cancel()
                break
        pool.shutdown(wait=True,cancel_futures=True)
    except BaseException as exc:
        terminate(); state["status"]="interrupted"; state["interrupt"]={"type":type(exc).__name__,"message":str(exc)}
        save(); pool.shutdown(wait=True,cancel_futures=True); raise
    state["status"]="failed" if state["failures"] else ("complete" if len(state["shards"])==len(indexes) else "partial")
    save(); return state

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition",choices=tuple(PARTITIONS),required=True)
    p.add_argument("--condition-index",action="append",type=int,help="stable global MAIN index 1..41; repeatable")
    p.add_argument("--workers",type=int); p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--resume",action="store_true")
    p.add_argument("--import-shard-dir",type=Path); p.add_argument("--wall-seconds",type=int,default=86400)
    p.add_argument("--deadline-utc"); p.add_argument("--probe",action="store_true")
    p.add_argument("--probe-receipt",type=Path,help="required for formal cloud partition; must match this dataset/model/reference")
    p.add_argument("--probe-reference",type=Path,default=DEFAULT_REFERENCE)
    return p.parse_args()

def main():
    args=parse_args()
    if args.wall_seconds<=0 or (args.resume and args.import_shard_dir):raise ValueError("invalid wall/resume/import flags")
    if args.probe and (args.resume or args.import_shard_dir or args.condition_index or args.probe_receipt):raise ValueError("probe cannot resume/import/select/use gate")
    indexes=tuple(args.condition_index) if args.condition_index else PARTITIONS[args.partition]
    if len(set(indexes))!=len(indexes) or any(i<1 or i>41 for i in indexes):raise ValueError("condition indexes must be unique 1..41")
    if args.condition_index and not set(indexes).issubset(PARTITIONS[args.partition]):raise ValueError("index outside selected partition")
    quota=cpu_quota(); cap=min(quota,4) if args.partition=="cloud" else quota
    requested=args.workers if args.workers is not None else (cap if args.partition=="cloud" else 1)
    if requested<=0:raise ValueError("--workers must be positive")
    workers=min(requested,cap)
    if workers!=requested:print(json.dumps({"event":"WORKERS_CLAMPED","requested":requested,"cpu_quota":quota,"using":workers}),flush=True)
    data=args.data.resolve(); data_sha=sha_file(data); gp=_gp_model_record(load_gp_rule())
    inventory=runtime_inventory(); episodes,_=actual_episodes(data)
    if len(episodes)!=840:raise ValueError("MAIN requires exactly 840 episodes")
    config=json.loads((ROOT/"configs/uwsaa.json").read_text(encoding="utf-8"))
    if config.get("dataset_sha256")!=data_sha: raise ValueError("uwsaa config dataset SHA differs from selected Caserta file")
    probe_sha=None
    if not args.probe and args.partition=="cloud":
        if args.probe_receipt is None: raise ValueError("formal cloud work requires a passing --probe-receipt")
        gate_path=args.probe_receipt.resolve()
        gate=json.loads(gate_path.read_text(encoding="utf-8"))
        if (gate.get("schema")!="gp_caserta_crosshost_probe_receipt_v1" or gate.get("status")!="pass"
            or gate.get("dataset_sha256")!=data_sha or gate.get("gp_model")!=gp
            or gate.get("reference_sha256")!=sha_file(args.probe_reference.resolve())
            or gate.get("runtime_inventory")!=inventory
            or gate.get("partition_runner_sha256")!=sha_file(Path(__file__))
            or gate.get("validation_script_sha256")!=sha_file(ROOT/"reproducibility/gp_arm_paired_analysis.py")
            or int(gate.get("matched_records",-1))!=252):
            raise ValueError("cloud probe gate does not match active dataset/model/reference")
        probe_sha=sha_file(gate_path)
    elif args.probe_receipt is not None:
        raise ValueError("--probe-receipt is only accepted for the formal cloud partition")
    setter=keep_awake()
    try:
        if args.probe:
            receipt=run_probe(args,data_sha,gp,episodes,workers,inventory); print(json.dumps(receipt,indent=2,ensure_ascii=False)); return 0
        state=run_partition(args,indexes,workers,data,data_sha,gp,inventory,probe_sha)
    finally:restore_awake(setter)
    print(json.dumps({"status":state["status"],"completed_conditions":len(state["completed_conditions"]),
        "expected_conditions":len(indexes),"completed_endpoint_rows":state["completed_endpoint_rows"],
        "failures":state["failures"],"run_manifest":str(args.output.resolve()/"run_manifest.json")},
        indent=2,ensure_ascii=False))
    return 0 if state["status"]=="complete" else 2 if state["status"]=="partial" else 1

if __name__=="__main__": raise SystemExit(main())