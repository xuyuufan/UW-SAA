"""Check preserved historical ablation endpoint coverage without reclassifying GP."""
from pathlib import Path
import csv,gzip,json,math,io,hashlib
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]/'results/historical-ablation-20261021'
def main():
    inventory=json.loads((ROOT/'source_inventory.json').read_text(encoding='utf-8'))
    counts=Counter();seen=set();solvers=Counter()
    index=json.loads((ROOT/'data/historical_ablation_endpoints.parts.json').read_text(encoding='utf-8'))
    buffer=io.BytesIO()
    for part in sorted(index['parts'],key=lambda p:p['sequence']):
        raw=(ROOT/'data'/part['filename']).read_bytes()
        assert len(raw)==part['bytes'] and hashlib.sha256(raw).hexdigest()==part['sha256']
        buffer.write(raw)
    assert hashlib.sha256(buffer.getbuffer()).hexdigest()==index['payload_sha256']
    buffer.seek(0)
    with gzip.GzipFile(fileobj=buffer) as gz,io.TextIOWrapper(gz,encoding='utf-8') as f:
        for r in csv.DictReader(f):
            assert r['evidence_status']=='accepted'
            assert r['source_id'] in inventory['row_counts_by_source']
            key=(r['source_id'],r['source_row']);assert key not in seen;seen.add(key)
            n=int(r['num_containers']);value=int(r['relocations'])
            assert n>0 and value>=0 and math.isclose(value/n,float(r['relocations_per_container']),abs_tol=1e-12)
            counts[r['source_id']]+=1;solvers[r['solver']]+=1
    assert dict(counts)==inventory['row_counts_by_source'] and sum(counts.values())==inventory['total_rows']==493920
    assert len(solvers)==6
    print(json.dumps({'status':'PASS_HISTORICAL_ABLATION','endpoint_rows':sum(counts.values()),
                      'counts_by_source':dict(counts),'GP_scope':'retired model; not replacement-GP ablation'}))
if __name__=='__main__':main()
