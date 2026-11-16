"""Additional fixed-setting sensitivity checks, using the shipped data only.

These tests were specified before their added runs, after the original study.
They are retrospective sensitivity analyses, not a new independent test set.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import __main__
if hasattr(__main__, 'cross_examine') and hasattr(__main__, 'STATIONS'):
    wa=__main__  # reuse notebook definitions without resetting its plotting backend
else:
    try:
        import weatheraudit as wa
    except ModuleNotFoundError:
        import WeatherAudit_run_v4 as wa  # local development entry

PROTOCOL = dict(
    version=1, seed=20261116, reporting_lags=[1, 8, 14, 30, 60],
    local_utc_offsets=dict(Quzhou=8, Shanghai=8, Hotan=8, Osaka=9,
                           Chuncheon=9, Gangneung=9, Pohang=9, Busan=9),
    station_label_shifts=[-1, 0, 1], calendar_bootstrap_draws=9999,
    comparison_family='8 rolling-vs-raw plus 8 boosting-vs-rolling',
    multiplicity='Holm, two-sided approximate centered calendar-week bootstrap p-values',
    signed_steps=[-2., -.5, .5, 2.], onset_fractions=[.35, .5, .65],
    null_trials_per_setting=50, null_ar_coefficients=[.3, .8], null_days=730,
    detector='original detector and thresholds, unchanged',
    status='additional retrospective checks; no settings selected from these results',
)


def _history(ctx):
    f=pd.concat([ctx['tr'],ctx['temp']]).sort_values('date').set_index('date')
    return f.reindex(pd.date_range(ctx['meta']['start'],ctx['meta']['end'],freq='D'))


def delayed_updates(ctx):
    f=_history(ctx)
    error=f.s_tmax-f.fc_tmax
    test=(f.index>=ctx['ts']) & f.s_tmax.notna() & f.fc_tmax.notna()
    rows=[]
    for lag in PROTOCOL['reporting_lags']:
        bias=error.shift(lag).rolling(30,min_periods=5).mean().fillna(0.)
        rows.append(dict(lag_days=lag,n=int(test.sum()),
                         raw_mae=round(float((f.fc_tmax-f.s_tmax)[test].abs().mean()),3),
                         rolling_mae=round(float((f.fc_tmax+bias-f.s_tmax)[test].abs().mean()),3)))
    # A frozen pre-test offset makes no use of any test observation.
    frozen=float(error[f.index<ctx['ts']].mean())
    return dict(rows=rows,no_test_update_mae=round(float(
        (f.fc_tmax+frozen-f.s_tmax)[test].abs().mean()),3),n=int(test.sum()))


def _hourly(ctx):
    start,end=ctx['meta']['start'],ctx['meta']['end']
    cols=['temperature_2m','temperature_2m_previous_day1','temperature_2m_previous_day2']
    frames=[]
    for year in range(start.year,end.year+1):
        params=dict(latitude=ctx['lat'],longitude=ctx['lon'],hourly=','.join(cols),
                    timezone='Asia/Shanghai',
                    start_date=max(f'{year}-01-01',start.strftime('%Y-%m-%d')),
                    end_date=min(f'{year}-12-31',end.strftime('%Y-%m-%d')))
        frame=pd.DataFrame(wa._get(wa.OM_PREV,params)['hourly'])
        frame['time']=pd.to_datetime(frame['time'])
        frames.append(frame)
    return pd.concat(frames).set_index('time').sort_index()


def day_window_sensitivity(ctx):
    hourly=_hourly(ctx)
    source='temperature_2m_previous_day1'
    full=_history(ctx)
    reference=full.s_tmax
    offset=PROTOCOL['local_utc_offsets'][ctx['name']]
    definitions=[('legacy_UTC+8',8,0),('station_local',offset,0)]
    definitions += [('local_shifted_station_labels',offset,s) for s in [-1,1]]
    aligned=[]
    for name,zone,shift in definitions:
        h=hourly[[source]].copy()
        h.index=h.index+pd.Timedelta(hours=zone-8)
        values=h[source].resample('D').max().where(h[source].resample('D').count()>=20)
        f=pd.DataFrame(dict(forecast=values,station=reference.shift(shift,freq='D')))
        f=f.reindex(pd.date_range(ctx['meta']['start'],ctx['meta']['end'],freq='D'))
        residual=f.station-f.forecast
        f['rolling']=f.forecast+residual.shift(8).rolling(30,min_periods=5).mean().fillna(0.)
        aligned.append((name,zone,shift,f))
    # Every row is compared on the same target dates, including the label perturbations.
    common=full.index[full.index>=ctx['ts']]
    for _,_,_,f in aligned:
        common=common.intersection(f.index[f.station.notna() & f.forecast.notna()])
    rows=[]
    for name,zone,shift,f in aligned:
        t=f.loc[common]
        rows.append(dict(definition=name,utc_offset_hours=zone,station_label_shift_days=shift,
                         n=len(t),raw_mae=round(float((t.forecast-t.station).abs().mean()),3),
                         rolling_mae=round(float((t['rolling']-t.station).abs().mean()),3)))
    return dict(rows=rows,warning='Local civil-day aggregation does not establish NOAA observation-day timing.')


def _calendar_bootstrap(difference,dates,seed):
    d=pd.Series(np.asarray(difference,float),index=pd.DatetimeIndex(dates)).dropna()
    groups=d.groupby(d.index.to_period('W-SUN')).agg(['sum','count'])
    sums=groups['sum'].to_numpy(); counts=groups['count'].to_numpy()
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(groups),size=(PROTOCOL['calendar_bootstrap_draws'],len(groups)))
    means=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    observed=float(d.mean())
    p=(1+int((np.abs(means-observed)>=abs(observed)).sum()))/(1+len(means))
    lo,hi=np.quantile(means,[.025,.975])
    return dict(gain=round(observed,4),lo=round(float(lo),4),hi=round(float(hi),4),
                p_approx=float(p),calendar_weeks=len(groups),n=len(d))


def multiple_comparisons(ctxs):
    rows=[]
    for i,(name,ctx) in enumerate(ctxs.items()):
        te=ctx['temp']; y=te.s_tmax
        raw=(te.fc_tmax-y).abs().to_numpy()
        roll=(te.fc_tmax+te.bias_30.fillna(0.)-y).abs().to_numpy()
        gb=(ctx['gb_pred']-y).abs().to_numpy()
        for label,values in [('rolling_vs_raw',raw-roll),('boosting_vs_rolling',roll-gb)]:
            rows.append(dict(station=name,comparison=label,
                             **_calendar_bootstrap(values,te.date,PROTOCOL['seed']+i)))
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_approx'])
    previous=0.
    for rank,i in enumerate(order):
        adjusted=min(1.,max(previous,(len(rows)-rank)*rows[i]['p_approx']))
        previous=adjusted
        rows[i]['p_holm']=round(adjusted,5)
        rows[i]['supported_after_holm']=bool(adjusted<=.05)
    return dict(family_size=len(rows),alpha=.05,rows=rows,
                qualification='Exploratory approximate resampling inference; not a distribution-free or prospective guarantee.')


def signed_injections():
    rows=[]
    for name,(sid,lat,lon) in wa.XE_SITES.items():
        days,S,F,E,_=wa.xe_frame(name,sid,lat,lon)
        keep=days<=np.datetime64(wa.XE_QUIET_END)
        days,S,F,E=days[keep],S[keep],F[keep],E[keep]
        if len(days)<2*wa.XE_MIN_SEG+60:
            continue
        control=[e for e in wa.cross_examine(days,S,F,E) if e['verdict']!='no step here']
        for fraction in PROTOCOL['onset_fractions']:
            at=days[int(len(days)*fraction)]
            baseline_near=any(abs(int((e['date']-at)/np.timedelta64(1,'D')))<=wa.XE_TOL_DAYS for e in control)
            for target in ['station','forecast','reanalysis']:
                for step in PROTOCOL['signed_steps']:
                    arrays=[v+np.where(days>=at,step,0.) if t==target else v
                            for t,v in zip(['station','forecast','reanalysis'],[S,F,E])]
                    events=wa.cross_examine(days,*arrays)
                    nearby=[e for e in events if e['verdict']!='no step here' and
                            abs(int((e['date']-at)/np.timedelta64(1,'D')))<=wa.XE_TOL_DAYS]
                    best=min(nearby,key=lambda e:abs(int((e['date']-at)/np.timedelta64(1,'D')))) if nearby else None
                    rows.append(dict(station=name,target=target,step_C=step,onset_fraction=fraction,
                                     injected_at=str(at),detected=bool(best),
                                     correct=bool(best and best['verdict']==target+' changed'),
                                     control_already_near=baseline_near,
                                     incremental_detected=bool(best and not baseline_near),
                                     verdict=best['verdict'] if best else None))
        print(f'[robustness] signed multi-onset injections: {name}',flush=True)
    summaries=[]
    for step in PROTOCOL['signed_steps']:
        group=[r for r in rows if r['step_C']==step]
        summaries.append(dict(step_C=step,trials=len(group),detected=sum(r['detected'] for r in group),
                              correct=sum(r['correct'] for r in group),
                              incremental_detected=sum(r['incremental_detected'] for r in group),
                              controls_already_near=sum(r['control_already_near'] for r in group)))
    return dict(rows=rows,summary=summaries,
                qualification='Real background may already contain an event; nearby control events are reported, not counted as newly recovered steps.')


def _wilson(hits,n):
    z=1.95996398454; p=hits/n; denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [round(center-half,4),round(center+half,4)]


def synthetic_null():
    rows=[]; n=PROTOCOL['null_days']
    days=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+n)
    truth=20+8*np.sin(np.arange(n)*2*np.pi/365.25)
    for setting,phi in enumerate(PROTOCOL['null_ar_coefficients']):
        for trial in range(PROTOCOL['null_trials_per_setting']):
            rng=np.random.default_rng(PROTOCOL['seed']+10000+1000*setting+trial)
            def ar(sd):
                noise=rng.normal(size=n); out=np.empty(n); out[0]=sd*noise[0]
                for j in range(1,n):out[j]=phi*out[j-1]+sd*math.sqrt(1-phi*phi)*noise[j]
                return out
            shared=ar(.8)
            S=truth+ar(.6); F=truth+shared+ar(.4); E=truth+.8*shared+ar(.4)
            events=[e for e in wa.cross_examine(days,S,F,E) if e['verdict']!='no step here']
            named=[e for e in events if e['verdict'].endswith(' changed')]
            rows.append(dict(phi=phi,trial=trial,events=len(events),named_false_events=len(named)))
        print(f'[robustness] known-null simulation: phi={phi}',flush=True)
    summary=[]
    for phi in PROTOCOL['null_ar_coefficients']:
        group=[r for r in rows if r['phi']==phi]; hits=sum(r['events']>0 for r in group)
        named=sum(r['named_false_events']>0 for r in group)
        summary.append(dict(phi=phi,trials=len(group),runs_with_any_false_event=hits,
                            runs_with_named_false_event=named,any_event_rate=hits/len(group),
                            any_event_rate_wilson95=_wilson(hits,len(group))))
    return dict(summary=summary,rows=rows,
                generator='730 days; stationary Gaussian AR(1) source errors, shared forecast/reanalysis component; no injected step or changing bias',
                qualification='False-event rates apply only to these specified simulations, not to real station records.')


def run_robustness(ctxs):
    Path('robustness_protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n',encoding='utf-8')
    result=dict(protocol=PROTOCOL,delay={},day_windows={})
    for name,ctx in ctxs.items():
        result['delay'][name]=delayed_updates(ctx)
        result['day_windows'][name]=day_window_sensitivity(ctx)
    result['multiplicity']=multiple_comparisons(ctxs)
    result['signed_injection']=signed_injections()
    result['synthetic_null']=synthetic_null()
    Path('robustness_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def print_robustness(result):
    print('\n=== Table 12. Longer publication-delay assumptions and frozen correction ===')
    wa.display(pd.DataFrame([dict(station=name,**r) for name,d in result['delay'].items() for r in d['rows']]))
    print('\n=== Table 13. Local day aggregation and +/-1 day label sensitivity: matched dates ===')
    wa.display(pd.DataFrame([dict(station=name,**r) for name,d in result['day_windows'].items() for r in d['rows']]))
    print('\n=== Table 14. Calendar-week bootstrap with Holm correction over 16 comparisons ===')
    wa.display(pd.DataFrame(result['multiplicity']['rows']))
    print('\n=== Table 15. Positive and negative steps at three onset dates ===')
    wa.display(pd.DataFrame(result['signed_injection']['summary']))
    print('\n=== Table 16. Known-null synthetic controls (rates apply only to this generator) ===')
    wa.display(pd.DataFrame(result['synthetic_null']['summary']))


CONSERVATIVE_PROTOCOL = dict(
    version=1, block_observations=28, permutations=999, alpha_per_pair=.05/3,
    search='one root split per pair; no recursive candidate searches',
    local_effective_block=28, minimum_jump_C=.5, minimum_t_like=2.,
    heldout_seed_base=20301116, trials_per_phi=50, ar_coefficients=[.3,.8],
    synthetic_step_C=[-2.,2.], synthetic_step_targets=['station','forecast','reanalysis'],
    step_trials_per_target=10, step_phi=.8, step_onset_fraction=.5,
    development='Specified once after inspecting original-detector failures; evaluated on different random seeds. No further tuning.',
    qualification='Conservative exploratory alternative, not a calibrated operational alarm or independent real-world validation.',
)


def conservative_cross_examine(days,S,F,E):
    """Longer dependence blocks, three root tests, and conservative local screening."""
    days=np.asarray(days,dtype='datetime64[D]')
    series={'S-F':np.asarray(S)-F,'S-E':np.asarray(S)-E,'F-E':np.asarray(F)-E}
    block=CONSERVATIVE_PROTOCOL['block_observations']
    candidates=[]
    for k,values in series.items():
        # Observed and permuted statistics use identical complete-block data.
        v=values[:len(values)//block*block]
        r=wa.xe_cusum(v)
        if r is None:continue
        i,stat=r
        p=wa.xe_perm_p(v,stat,block=block,n_iter=CONSERVATIVE_PROTOCOL['permutations'],seed=wa.XE_SEED[k])
        if p<=CONSERVATIVE_PROTOCOL['alpha_per_pair']:candidates.append(days[i])
    events=[]
    for date in wa.xe_cluster(candidates):
        local={}
        for k,values in series.items():
            step=wa.xe_local_step(days,values,date)
            local[k]=(step[0],step[1]*math.sqrt(wa.XE_BLOCK/block),step[2],step[3]) if step else None
        verdict=wa.xe_verdict(local)
        if verdict!='no step here':events.append(dict(date=date,verdict=verdict,steps=local))
    return events


def _simulation(seed,phi,n=730):
    rng=np.random.default_rng(seed)
    days=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+n)
    truth=20+8*np.sin(np.arange(n)*2*np.pi/365.25)
    def ar(sd):
        noise=rng.normal(size=n); out=np.empty(n); out[0]=sd*noise[0]
        for j in range(1,n):out[j]=phi*out[j-1]+sd*math.sqrt(1-phi*phi)*noise[j]
        return out
    shared=ar(.8)
    return days,truth+ar(.6),truth+shared+ar(.4),truth+.8*shared+ar(.4)


def run_conservative_checks():
    """Compare versions on new null seeds, known synthetic steps, and original flags."""
    p=CONSERVATIVE_PROTOCOL
    original_seeds={PROTOCOL['seed']+10000+1000*s+t for s in range(2)
                    for t in range(PROTOCOL['null_trials_per_setting'])}
    new_seeds={p['heldout_seed_base']+1000*s+t for s in range(2)
               for t in range(p['trials_per_phi'])}
    assert original_seeds.isdisjoint(new_seeds), 'Validation seeds overlap development simulations'
    Path('detector_improvement_protocol.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
    detectors={'original':wa.cross_examine,'conservative':conservative_cross_examine}
    null=[]; steps=[]; provenance=[]
    for setting,phi in enumerate(p['ar_coefficients']):
        for trial in range(p['trials_per_phi']):
            data=_simulation(p['heldout_seed_base']+setting*1000+trial,phi)
            for label,detector in detectors.items():
                ev=[e for e in detector(*data) if e['verdict']!='no step here']
                null.append(dict(detector=label,phi=phi,trial=trial,events=len(ev),
                                 named=sum(e['verdict'].endswith(' changed') for e in ev)))
        print(f'[improved detector] held-out synthetic null phi={phi}',flush=True)
    for trial in range(p['step_trials_per_target']):
        days,S,F,E=_simulation(p['heldout_seed_base']+5000+trial,p['step_phi'])
        at=days[int(len(days)*p['step_onset_fraction'])]
        for target in p['synthetic_step_targets']:
            for step in p['synthetic_step_C']:
                arrays=[v+np.where(days>=at,step,0.) if t==target else v
                        for t,v in zip(p['synthetic_step_targets'],[S,F,E])]
                for label,detector in detectors.items():
                    near=[e for e in detector(days,*arrays) if e['verdict']!='no step here' and
                          abs(int((e['date']-at)/np.timedelta64(1,'D')))<=wa.XE_TOL_DAYS]
                    best=min(near,key=lambda e:abs(int((e['date']-at)/np.timedelta64(1,'D')))) if near else None
                    steps.append(dict(detector=label,target=target,step_C=step,trial=trial,
                                      detected=bool(best),correct=bool(best and best['verdict']==target+' changed')))
    for name,(sid,lat,lon) in wa.XE_SITES.items():
        days,S,F,E,flags=wa.xe_frame(name,sid,lat,lon)
        switches=wa.xe_flag_switches(days,flags)
        ev=conservative_cross_examine(days,S,F,E)
        provenance.append(dict(station=name,scored=wa.xe_score(ev,switches),
                               events=[dict(date=str(e['date']),verdict=e['verdict']) for e in ev]))
    summaries=[]
    for phi in p['ar_coefficients']:
        for label in detectors:
            group=[r for r in null if r['phi']==phi and r['detector']==label]
            hits=sum(r['events']>0 for r in group); named=sum(r['named']>0 for r in group)
            summaries.append(dict(detector=label,phi=phi,trials=len(group),false_event_runs=hits,
                                  named_false_event_runs=named,rate=hits/len(group),wilson95=_wilson(hits,len(group))))
    step_summary=[]
    for label in detectors:
        for step in p['synthetic_step_C']:
            group=[r for r in steps if r['detector']==label and r['step_C']==step]
            step_summary.append(dict(detector=label,step_C=step,trials=len(group),
                                     detected=sum(r['detected'] for r in group),correct=sum(r['correct'] for r in group)))
    result=dict(protocol=p,null_summary=summaries,null_rows=null,step_summary=step_summary,
                step_rows=steps,provenance=provenance)
    Path('detector_improvement_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def print_conservative_checks(result):
    print('\n=== Table 17. Conservative detector versus original: new synthetic null seeds ===')
    wa.display(pd.DataFrame(result['null_summary']))
    print('\n=== Table 18. Synthetic +/-2 C steps under strong autocorrelation: detection tradeoff ===')
    wa.display(pd.DataFrame(result['step_summary']))
    print('\n=== Table 19. Conservative detector: same retrospective provenance answer key ===')
    wa.display(pd.DataFrame([dict(station=r['station'],**s) for r in result['provenance'] for s in r['scored']]))
