"""Verify shipped file integrity and key reporting claims, without running models."""
from pathlib import Path
import base64
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def main():
    cache_meta=json.loads((ROOT/'cache_manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(ROOT/'cache.zip') as archive:
        require(archive.testzip() is None,'Cache archive CRC failed')
        require(set(archive.namelist())==set(cache_meta['files']),'Cache inventory differs')
        for name,info in cache_meta['files'].items():
            payload=archive.read(name)
            require(len(payload)==info['bytes'],f'Cache length differs: {name}')
            require(hashlib.sha256(payload).hexdigest()==info['sha256'],f'Cache hash differs: {name}')
    nb=json.loads((ROOT/'WeatherAudit.ipynb').read_text(encoding='utf-8'))
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    outputs=[o for c in code for o in c.get('outputs',[])]
    require(len(nb['cells'])==24 and len(code)==16,'Unexpected notebook structure')
    require([c['execution_count'] for c in code]==list(range(1,17)),'Notebook is not fully executed in order')
    require(not any(o.get('output_type')=='error' for o in outputs),'Notebook contains an error')
    images=[o['data']['image/png'] for o in outputs if 'image/png' in o.get('data',{})]
    require(len(images)==10,'Expected ten embedded figures')
    for value in images:
        payload=base64.b64decode(''.join(value) if isinstance(value,list) else value)
        require(payload.startswith(b'\x89PNG\r\n\x1a\n'),'Invalid embedded PNG')

    script=(ROOT/'weatheraudit.py').read_text(encoding='utf-8')
    compile(script,'weatheraudit.py','exec')
    parts=re.split(r'^\s*# %% CELL: (\w+)\s*$',script,flags=re.M)
    cells={'imports':parts[0].rstrip()}
    for i in range(1,len(parts)-1,2):cells[parts[i]]=parts[i+1].strip()
    names=['imports','config','fetch','frames','models','climate','multimodel','uncertainty',
           'smartbase','climatology','fairness','crossexam','run','summary','figures','html']
    for c,name in zip(code,names):
        source=cells[name]
        if name in ('run','summary','figures','html'):
            source=source.replace('if __name__ == "__main__":','')
            source='\n'.join((ln[4:] if ln.startswith('    ') else ln) if ln.strip() else '' for ln in source.splitlines()).strip()
        if name=='imports':source=source.replace('matplotlib.use("Agg")\n','')
        if name=='html':source+='\n\ndisplay(HTML(html))'
        require(source.strip()==''.join(c['source']).strip(),f'Script and notebook differ: {name}')

    d=json.loads((ROOT/'results_v4.json').read_text(encoding='utf-8'))
    require(len(d['cities'])==8,'Expected eight station audits')
    expected={
        'Quzhou':(1.86,1.20,1.31),'Shanghai':(.98,.86,.94),'Osaka':(1.52,.68,.65),
        'Hotan':(1.18,.92,1.05),'Chuncheon':(1.58,1.12,1.08),'Gangneung':(1.25,1.02,1.03),
        'Pohang':(1.70,.96,.84),'Busan':(1.90,1.05,1.01)}
    for city,(raw,rolling,gb) in expected.items():
        f=d['fair'][city]; result=d['cities'][city]
        require((round(result['raw_mae'],2),round(f['mae']['roll30'],2),round(result['gb_mae'],2))==(raw,rolling,gb),f'Temperature headline differs: {city}')
        require(f['roll30_vs_raw']['lo']>0,f'Rolling gain interval includes zero: {city}')
        for reference in ['yesterday','archive_lag_8_days']:
            p=f['persistence'][reference]
            require(0<p['n']<=p['test_days'] and p['roll30_mae_same_days']<p['persistence_mae'],f'Persistence comparison invalid: {city}')
    gains={city for city,f in d['fair'].items() if f['gbm_vs_roll30']['lo']>0}
    require(gains=={'Osaka','Pohang'},'Significant boosting gains differ')
    require(sum(c['brier_gb']<c['brier_raw'] for c in d['cities'].values())==7,'Rain-vs-forecast count differs')
    require(sum(c['brier_gb']<c['brier_base'] for c in d['cities'].values())==6,'Rain-vs-constant count differs')
    require(d['cities']['Quzhou']['cost_gb_cal']==610,'Updated Quzhou umbrella cost differs')
    require(d['lead_time']['Quzhou']['day1']['raw_mae']==2.06,'Day-ahead baseline differs')
    require(round(d['lead_time']['Quzhou']['day1']['roll30_mae'],2)==1.48,'Day-ahead correction differs')
    keys=[k for station in d['crossexam'].values() for k in station['answer_key']]
    require(len(keys)==4 and sum(k['correct'] for k in keys)==3,'Provenance match count differs')
    require(sum(v['injection']['control_events'] for v in d['crossexam'].values() if v['injection'])==12,'Control event count differs')
    rows2=[r for v in d['crossexam'].values() if v['injection'] for r in v['injection']['rows'] if r['size']==2]
    require((len(rows2),sum(r['detected'] for r in rows2),sum(r['correct'] for r in rows2))==(9,8,6),'Positive injection result differs')
    coverage=[v['coverage'] for v in d['adaptive_interval'].values()]
    require(round(min(coverage)*100)==76 and round(max(coverage)*100)==81,'Empirical interval range differs')

    r=json.loads((ROOT/'robustness_results.json').read_text())
    p=json.loads((ROOT/'robustness_protocol.json').read_text())
    require(r['protocol']==p,'Sensitivity protocol and results differ')
    require(set(r['delay'])==set(expected),'Sensitivity station inventory differs')
    for city,record in r['delay'].items():
        require([v['lag_days'] for v in record['rows']]==[1,8,14,30,60],f'Delay settings differ: {city}')
        require(all(v['n']==record['n'] and v['rolling_mae']<v['raw_mae'] for v in record['rows']),f'Delay claim differs: {city}')
        windows=r['day_windows'][city]['rows']
        require(len({v['n'] for v in windows})==1 and len(windows)==4,f'Day-window dates differ: {city}')
    holm=r['multiplicity']['rows']
    require(len(holm)==16 and all(0<v['p_approx']<=v['p_holm']<=1 for v in holm),'Invalid multiplicity results')
    require(sum(v['supported_after_holm'] and v['gain']>0 for v in holm if v['comparison']=='rolling_vs_raw')==7,'Adjusted rolling gain count differs')
    require(not any(v['supported_after_holm'] and v['gain']>0 for v in holm if v['comparison']=='boosting_vs_rolling'),'Adjusted boosting claim differs')
    require(len(r['signed_injection']['rows'])==108,'Signed injection trials incomplete')
    for summary in r['signed_injection']['summary']:
        rows=[v for v in r['signed_injection']['rows'] if v['step_C']==summary['step_C']]
        require(summary['trials']==len(rows)==27,'Signed injection group incomplete')
        require(summary['incremental_detected']==sum(v['detected'] and not v['control_already_near'] for v in rows),'Control-overlap accounting differs')

    c=json.loads((ROOT/'detector_improvement_results.json').read_text())
    cp=json.loads((ROOT/'detector_improvement_protocol.json').read_text())
    require(c['protocol']==cp,'Conservative protocol and results differ')
    old_seeds={p['seed']+10000+1000*s+t for s in range(2) for t in range(p['null_trials_per_setting'])}
    new_seeds={cp['heldout_seed_base']+1000*s+t for s in range(2) for t in range(cp['trials_per_phi'])}
    require(old_seeds.isdisjoint(new_seeds),'Simulation seeds overlap development batch')
    strong={v['detector']:v for v in c['null_summary'] if v['phi']==.8}
    require(strong['original']['false_event_runs']==20 and strong['conservative']['false_event_runs']==0,'Strong-null improvement claim differs')
    require(strong['conservative']['trials']==50 and strong['conservative']['wilson95'][1]==.0713,'Finite-simulation uncertainty differs')
    for row in c['null_summary']:
        trials=[v for v in c['null_rows'] if v['detector']==row['detector'] and v['phi']==row['phi']]
        require(len(trials)==row['trials'] and sum(v['events']>0 for v in trials)==row['false_event_runs'],'Null summary differs from trials')
    step=[v for v in c['step_rows'] if v['detector']=='conservative']
    require(len(step)==60 and all(v['correct'] for v in step),'Conservative large-step claim differs')
    conservative_keys=[s for v in c['provenance'] for s in v['scored']]
    require(len(conservative_keys)==4 and sum(v['correct'] for v in conservative_keys)==2,'Conservative real-switch tradeoff differs')

    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(ROOT/'WeatherAudit_writeup.docx') as archive:
        document=ET.fromstring(archive.read('word/document.xml'))
        tables=document.findall('.//w:tbl',ns)
        require(len(tables)==1,'One-page write-up table missing')
        table_rows=[]
        for row in tables[0].findall('w:tr',ns):
            table_rows.append([''.join(n.itertext()) for n in row.findall('w:tc',ns)])
        for row in table_rows[1:]:
            require(tuple(map(float,row[1:]))==expected[row[0]],f'Word table differs: {row[0]}')
        core=archive.read('docProps/core.xml').decode()
        require('Administrator' not in core,'Personal metadata in Word source')
    for path in ROOT.glob('*.md'):
        require('\ufffd' not in path.read_text(encoding='utf-8'),f'Encoding damage: {path.name}')
    for path in ROOT.glob('*.py'):
        compile(path.read_text(encoding='utf-8'),path.name,'exec')
    require((ROOT/'WeatherAudit_writeup.pdf').read_bytes().startswith(b'%PDF-'),'Invalid PDF')
    print(json.dumps(dict(status='PASS',cache_files=len(cache_meta['files']),cells=24,
                         outputs=len(outputs),images=len(images),errors=0,stations=8,
                         provenance_matches='original 3/4; conservative 2/4 within 45 days',
                         unadjusted_boosting_gains=sorted(gains),holm_supported_rolling_gains=7,
                         holm_supported_boosting_gains=0,strong_null_false_event_runs='20/50 -> 0/50',
                         conservative_synthetic_step_attribution='60/60'),indent=2))


if __name__=='__main__':
    main()
