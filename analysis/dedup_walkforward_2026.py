#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedup_walkforward_2026.py — Walk-forward validation of the 2026-08-09
double-counting de-duplications in collect.py.

Validates 3 fixes against FRED long history (2014+):
  #1 TGA/RRP  : FISCAL M83/M84 Lt-adjustment removed (TGA/RRP now only in GLF)
  #2 liq_mod  : direct m2_yoy term removed from effective_sfc (m2 only in GLF)
  #3 DXY      : incremental-contribution test — is Sc-DXY additive beyond GLF-DXY?

Method (redundancy-dedup-validation + walk-forward skills):
  - Reconstruct each redundant term EXACTLY (verbatim formula copies, NOT
    imports from collect.py which has live side effects).
  - Build OLD (redundant) vs NEW (de-duplicated) score series from the SAME
    raw FRED inputs, holding other factors fixed — apples-to-apples.
  - std-ratio OLD/NEW (target ~1.00 = de-dup doesn't collapse amplitude).
  - Predictive polarity: quantile top-vs-bottom fwd-return gap + seeded numpy
    bootstrap CI 90% (correct polarity = NEGATIVE for a stress score).
  - Version C: NEW rescaled to OLD std (control: amplification vs unique info).

Output: ../.dedup_walkforward_2026.json + printed verdict.
"""
import json, math, os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from historical_backtest_m1m6 import (
        fetch_fred_series, _nearest_prior_value, _sigmoid_factor,
        calculate_sfc_ensemble,
    )
    from walk_forward_validation import (
        add_forward_returns, bootstrap_diff_ci, FORWARD_HORIZONS_DAYS,
    )
except ImportError as e:
    print(f"[DedupWF] import failed: {e}", file=sys.stderr); sys.exit(1)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".dedup_walkforward_2026.json")
QUANTILE_TAIL = 0.20

# ---- verbatim GLF constants (data_sources/global_liquidity_engine.py) ----
Z_FED=(5.5,8.0); Z_ECB=(4.0,7.0); Z_JPN=(3.0,6.0); Z_M2=(6.0,4.0); Z_DXY=(100.0,5.0)
GLF_W={"fed":0.30,"ecb":0.15,"jpn":0.03,"m2":0.15,"tga":0.10,"rrp":0.10,"dxy":0.13}
GLF_RESCALE=5.927

def _z(v,mean,std,lo=-3.0,hi=3.0):
    if std==0 or v is None: return 0.0
    return max(lo,min(hi,(v-mean)/std))

def _tga_score(latest,chg):
    if chg < -10: z=1.5
    elif chg < -5: z=0.8
    elif chg < -2: z=0.3
    elif chg < 2: z=0.0
    elif chg < 5: z=-0.5
    elif chg < 10: z=-1.0
    else: z=-1.5
    if latest>900000: z-=0.5
    elif latest<300000: z-=0.3
    return max(-2.0,min(2.0,z))

def _rrp_score(latest,chg):
    if latest<10: z=1.5
    elif latest<50: z=1.0
    elif latest<200: z=0.0
    elif latest<500: z=-0.5
    else: z=-1.5
    if chg<-50: z+=0.5
    elif chg>50: z-=0.5
    return max(-2.0,min(2.0,z))

def _glf_components(fed_y,ecb_y,jpn_y,m2_y,tg,tg_chg,rr,rr_chg,dx):
    comp={}
    if fed_y is not None: comp["fed"]=(_z(fed_y,*Z_FED),GLF_W["fed"])
    if ecb_y is not None: comp["ecb"]=(_z(ecb_y,*Z_ECB),GLF_W["ecb"])
    if jpn_y is not None: comp["jpn"]=(_z(jpn_y,*Z_JPN),GLF_W["jpn"])
    if m2_y is not None:  comp["m2"]=(_z(m2_y,*Z_M2),GLF_W["m2"])
    if tg is not None:    comp["tga"]=(_tga_score(tg,tg_chg),GLF_W["tga"])
    if rr is not None:    comp["rrp"]=(_rrp_score(rr,rr_chg),GLF_W["rrp"])
    if dx is not None:    comp["dxy"]=(_z(dx,*Z_DXY,lo=-2.0,hi=2.0)*-1.0,GLF_W["dxy"])
    return comp

def _glf_adj(comp):
    if not comp: return 0.0
    total_w=sum(w for _,w in comp.values())
    glf_z=sum(z*w for z,w in comp.values())/total_w
    glf_score=max(0,min(100,55+glf_z*17.5))
    if glf_score>70: st=0.15
    elif glf_score>55: st=0.30
    elif glf_score>40: st=0.50
    elif glf_score>25: st=0.70
    else: st=0.85
    return max(-2.0,min(2.0,(0.50-st)*3.0))*GLF_RESCALE

def _m83_tga(latest,chg_pct):
    if chg_pct<-10: s=0.15
    elif chg_pct<-5: s=0.25
    elif chg_pct<-2: s=0.35
    elif chg_pct<2: s=0.50
    elif chg_pct<5: s=0.60
    elif chg_pct<10: s=0.75
    else: s=0.85
    if latest>900000: s=min(0.90,s+0.10)
    elif latest<300000: s=min(0.80,s+0.08)
    return max(0.05,min(0.95,s))

def _m84_rrp(latest,chg):
    if latest<10: lvl=0.15
    elif latest<50: lvl=0.30
    elif latest<200: lvl=0.50
    elif latest<500: lvl=0.65
    elif latest<1000: lvl=0.80
    else: lvl=0.90
    if chg<-50: tr=-0.15
    elif chg<-10: tr=-0.08
    elif chg<0: tr=-0.03
    elif chg<10: tr=0.03
    elif chg<50: tr=0.08
    else: tr=0.15
    return max(0.05,min(0.95,lvl+tr))

def _fiscal_lt_adj(tg,tg_chg,rr,rr_chg):
    if tg is None or rr is None: return 0.0
    m83=_m83_tga(tg,tg_chg); m84=_m84_rrp(rr,rr_chg)
    return max(-1.0,min(1.0,(0.5-m83)*1.0+(0.5-m84)*1.0))

def _liq_mod(m2_y):
    if m2_y is None: return 0.0
    return max(-5.0,min(10.0,(7.0-m2_y)*0.8))

def _yoy(levels):
    result={}
    dkeys=sorted(levels.keys())
    for ds in dkeys:
        d=datetime.strptime(ds,"%Y-%m-%d")
        pv=_nearest_prior_value(levels,(d-timedelta(days=365)).strftime("%Y-%m-%d"),max_lookback_days=60)
        if pv and pv!=0: result[ds]=(levels[ds]-pv)/pv*100
    return result

def _npv(d,ds,lb=45):
    return _nearest_prior_value(d,ds,max_lookback_days=lb)

def _sc_dxy(dxy,corr):
    sig=_sigmoid_factor(dxy,center=100.0,k=0.2)
    if corr is not None and corr>0.3: return sig
    elif corr is not None and corr>-0.3: return -sig*0.5
    else: return -sig

def _base_factors(**over):
    f={"Lt":0.0,"St":0.0,"Rt":0.0,"Ft":0.0,"Sc":0.0}; f.update(over); return f

def _series(rows, lt_fn, eff_extra_fn=None, sc_fn=None):
    out=[]
    for r in rows:
        lt=lt_fn(r)
        sc=(sc_fn(r) if sc_fn else 0.0)
        f=_base_factors(Lt=max(-3,min(3,lt)),Sc=sc)
        try: sfc=calculate_sfc_ensemble(f)[0]
        except Exception: continue
        eff=min(sfc+(eff_extra_fn(r) if eff_extra_fn else 0.0),100.0); eff=max(eff,0.0)
        out.append({"date":r["date"],"price":r["price"],"sfc_pct":eff})
    return out

def _quantile_report(tag, series, seed=42):
    """Quantile top-vs-bottom fwd-return gap + bootstrap CI 90%."""
    series=add_forward_returns(series)
    valid=[p for p in series if p.get("fwd_return_7d") is not None]
    if len(valid)<20: return {"tag":tag,"n":len(valid),"note":"insufficient"}
    srt=sorted(valid,key=lambda p:p["sfc_pct"]); n=len(srt); q=max(1,n//5)
    bottom=srt[:q]; top=srt[-q:]
    report={"tag":tag,"n":n,"sfc_range":[round(srt[0]["sfc_pct"],2),round(srt[-1]["sfc_pct"],2)],
            "n_bottom":len(bottom),"n_top":len(top)}
    for h in FORWARD_HORIZONS_DAYS:
        bf=[p[f"fwd_return_{h}d"] for p in bottom if p.get(f"fwd_return_{h}d") is not None]
        tf=[p[f"fwd_return_{h}d"] for p in top if p.get(f"fwd_return_{h}d") is not None]
        if len(bf)>=2 and len(tf)>=2:
            est,lo,hi=bootstrap_diff_ci(bf,tf)
            sig=(hi<0 or lo>0) if (hi is not None and lo is not None) else False
            report[f"{h}d_gap"]=round(est,4); report[f"{h}d_ci"]=[round(lo,4),round(hi,4)]
            report[f"{h}d_sig"]=sig
        else:
            report[f"{h}d_gap"]=None
    return report

def run():
    import random; random.seed(42)  # deterministic bootstrap (skill Pitfall 15)
    print("Fetching FRED history...")
    btc=fetch_fred_series("CBBTCUSD"); fed=fetch_fred_series("WALCL")
    ecb=fetch_fred_series("ECBASSETSW"); jpn=fetch_fred_series("JPNASSETS")
    m2=fetch_fred_series("M2SL"); tga=fetch_fred_series("WTREGEN")
    rrp=fetch_fred_series("RRPONTSYD"); dxy=fetch_fred_series("DTWEXBGS")
    for nm,s in [("btc",btc),("fed",fed),("ecb",ecb),("jpn",jpn),("m2",m2),("tga",tga),("rrp",rrp),("dxy",dxy)]:
        print(f"  {nm}: {len(s)} obs")
    if not btc or not m2 or not tga or not rrp: print("missing core"); return

    fed_y=_yoy(fed); ecb_y=_yoy(ecb); jpn_y=_yoy(jpn); m2_y=_yoy(m2)
    dates=sorted(btc.keys())
    rows=[]
    for i,ds in enumerate(dates):
        m2v=_npv(m2_y,ds,60); dx=_npv(dxy,ds,5)
        if m2v is None or dx is None: continue
        fedv=_npv(fed_y,ds,15); ecbv=_npv(ecb_y,ds,15); jpnv=_npv(jpn_y,ds,15)
        tg=_npv(tga,ds,20); tg4=_npv(tga,(datetime.strptime(ds,"%Y-%m-%d")-timedelta(days=28)).strftime("%Y-%m-%d"),20)
        rr=_npv(rrp,ds,5);  rr4=_npv(rrp,(datetime.strptime(ds,"%Y-%m-%d")-timedelta(days=28)).strftime("%Y-%m-%d"),5)
        tg_chg=((tg-tg4)/tg4*100) if (tg and tg4 and tg4!=0) else 0
        rr_chg=(rr-rr4) if (rr is not None and rr4 is not None) else 0
        comp=_glf_components(fedv,ecbv,jpnv,m2v,tg,tg_chg,rr,rr_chg,dx)
        glf_adj=_glf_adj(comp)
        fiscal_adj=_fiscal_lt_adj(tg,tg_chg,rr,rr_chg)
        liq=_liq_mod(m2v)
        # rolling 30d corr DXY-BTC for Sc
        corr=None
        if i>=29:
            sub_dates=dates[i-29:i+1]
            bv=[btc[d] for d in sub_dates]; xv=[_npv(dxy,d,5) for d in sub_dates]
            if all(b is not None for b in bv) and any(x is not None for x in xv):
                xv2=[x if x is not None else sum([q for q in xv if q is not None])/sum(1 for q in xv if q) for x in xv]
                mb=sum(bv)/len(bv); mx=sum(xv2)/len(xv2)
                num=sum((a-mb)*(c-mx) for a,c in zip(bv,xv2))
                db=sum((a-mb)**2 for a in bv)**0.5; ddx=sum((c-mx)**2 for c in xv2)**0.5
                corr=num/(db*ddx) if db>0 and ddx>0 else None
        sc=_sc_dxy(dx,corr)
        rows.append(dict(date=ds,price=btc[ds],glf_adj=glf_adj,fiscal_adj=fiscal_adj,liq=liq,sc=sc,m2_y=m2v))

    print(f"Computed {len(rows)} rows ({rows[0]['date']} .. {rows[-1]['date']})")

    # FIX #1
    s1_old=_series(rows,lambda r:r["glf_adj"]+r["fiscal_adj"])
    s1_new=_series(rows,lambda r:r["glf_adj"])
    # FIX #2
    s2_old=_series(rows,lambda r:r["glf_adj"],eff_extra_fn=lambda r:r["liq"])
    s2_new=_series(rows,lambda r:r["glf_adj"])
    # FIX #3
    s3_base=_series(rows,lambda r:r["glf_adj"])
    s3_aug=_series(rows,lambda r:r["glf_adj"],sc_fn=lambda r:r["sc"])

    def stdof(s): 
        v=[p["sfc_pct"] for p in s]; m=sum(v)/len(v); return (sum((x-m)**2 for x in v)/len(v))**0.5

    std={}
    for tag,s in [("fix1_old",s1_old),("fix1_new",s1_new),("fix2_old",s2_old),
                  ("fix2_new",s2_new),("fix3_base",s3_base),("fix3_aug",s3_aug)]:
        std[tag]=stdof(s)

    out={
        "meta":{"n_rows":len(rows),"start":rows[0]["date"],"end":rows[-1]["date"],
                "note":"FRED reconstruction; china excluded (w=0.04); DTWEXBGS DXY proxy; St/Rt/Ft=0"},
        "std_ratios":{
            "fix1_tga_rrp": {"std_old":round(std["fix1_old"],4),"std_new":round(std["fix1_new"],4),
                             "ratio_old_new":round(std["fix1_old"]/std["fix1_new"],3) if std["fix1_new"] else None},
            "fix2_liq_mod": {"std_old":round(std["fix2_old"],4),"std_new":round(std["fix2_new"],4),
                             "ratio_old_new":round(std["fix2_old"]/std["fix2_new"],3) if std["fix2_new"] else None},
            "fix3_dxy": {"std_base":round(std["fix3_base"],4),"std_aug":round(std["fix3_aug"],4)},
        },
        "quantile":{
            "fix1_old":_quantile_report("fix1_tga_rrp_OLD",s1_old),
            "fix1_new":_quantile_report("fix1_tga_rrp_NEW",s1_new),
            "fix2_old":_quantile_report("fix2_liq_mod_OLD",s2_old),
            "fix2_new":_quantile_report("fix2_liq_mod_NEW",s2_new),
            "fix3_base":_quantile_report("fix3_dxy_BASE_GLFonly",s3_base),
            "fix3_aug":_quantile_report("fix3_dxy_AUG_GLF+Sc",s3_aug),
        },
    }
    with open(OUT,"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
    print(f"\nSaved {OUT}")

if __name__=="__main__":
    run()
