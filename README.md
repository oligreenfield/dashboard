# dashboard

"""
Inflation desk dashboard -> self-contained HTML (Plotly.js via CDN), shown inline in Jupyter.
Tabs: RPI swaps | HICP swaps | GBP linkers | EUR linkers
Run:  exec(open("inflation_dashboard.py").read())   or paste into a cell.
"""
import re, json, numpy as np, pandas as pd
from IPython.display import IFrame, display

# =====================================================================
# 1) >>> INPUT YOUR DATA HERE <<<
# Each df: index = dates (daily), columns = tenors/maturities ("2Y","5Y","10Y"... or "2029","2035"...),
# values = swap rates / real yields (same units across the df).
# =====================================================================
USE_DEMO = True          # set False once your data is loaded below

if not USE_DEMO:
    rpi_swaps  = pd.read_csv("rpi_swaps.csv",  index_col=0, parse_dates=True)
    hicp_swaps = pd.read_csv("hicp_swaps.csv", index_col=0, parse_dates=True)
    gbp_real   = pd.read_csv("gbp_real.csv",   index_col=0, parse_dates=True)
    eur_real   = pd.read_csv("eur_real.csv",   index_col=0, parse_dates=True)
else:  # synthetic data just to test the dashboard
    def _demo(tenors, base, seed):
        rng = np.random.default_rng(seed); n = 1000
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        lvl = np.cumsum(rng.normal(0, .03, n)); slp = np.cumsum(rng.normal(0, .015, n))
        x = np.array([float(re.sub(r"[^\d.]", "", t)) for t in tenors]); x = (x - x.mean()) / x.std()
        y = base + lvl[:, None] + slp[:, None] * x[None] + rng.normal(0, .02, (n, len(x)))
        return pd.DataFrame(y, idx, tenors)
    _t = ["2Y","3Y","5Y","7Y","10Y","15Y","20Y","30Y"]
    rpi_swaps, hicp_swaps = _demo(_t, 3.4, 1), _demo(_t, 2.3, 2)
    gbp_real, eur_real = _demo(["2Y","4Y","6Y","8Y","10Y","15Y","25Y","40Y"], 0.8, 3), _demo(_t, 0.2, 4)

# unit = multiplier converting the data to bp for fly/curve tables (100 if quoted in %, 10000 if decimals, 1 if bp)
# zero = True for swap tabs (adds the implied forward curve panel)
CONFIG = {
    "RPI swaps":   dict(df=rpi_swaps,  unit=100, zero=True),
    "HICP swaps":  dict(df=hicp_swaps, unit=100, zero=True),
    "GBP linkers": dict(df=gbp_real,   unit=100, zero=False),
    "EUR linkers": dict(df=eur_real,   unit=100, zero=False),
}
N_PC = 3            # number of principal components computed
PCA_WINDOW = None   # None = PCA on full history of daily changes; or e.g. 750 = last 750 days
OUTFILE = "inflation_dashboard.html"

# =====================================================================
# 2) PCA prep (Python side)
# =====================================================================
def _tn(c):
    m = re.search(r"[\d.]+", c); v = float(m.group()) if m else 0.0
    return v / 12 if c.upper().strip().endswith("M") else v

def prep(df, unit, zero):
    df = df.copy(); df.columns = [str(c) for c in df.columns]
    df = df.sort_index().ffill().dropna()
    df = df[sorted(df.columns, key=_tn)]
    X = df.values.astype(float); dX = np.diff(X, axis=0)
    if PCA_WINDOW: dX = dX[-PCA_WINDOW:]
    w, v = np.linalg.eigh(np.cov(dX.T)); o = np.argsort(w)[::-1]; w, v = w[o], v[:, o]
    L = v[:, :N_PC].copy(); m = len(df.columns)
    for k in range(L.shape[1]):                      # sign conventions: PC1 level+, PC2 slope+, PC3 belly+
        s = L[:, k].sum() if k == 0 else (L[-1, k] - L[0, k] if k == 1 else L[m // 2, k])
        if s < 0: L[:, k] *= -1
    mu = X.mean(0); S = (X - mu) @ L
    return dict(dates=[d.strftime("%Y-%m-%d") for d in df.index], tenors=list(df.columns),
                tn=[_tn(c) for c in df.columns], data=X.T.round(5).tolist(), L=L.round(5).tolist(),
                S=S.round(5).tolist(), mean=mu.round(5).tolist(), var=(w[:N_PC] / w.sum()).round(4).tolist(),
                unit=unit, zero=zero)

# =====================================================================
# 3) HTML / JS front end
# =====================================================================
TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Inflation dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:0;padding:10px;font-size:13px;color:#222}
.tabs button{padding:8px 16px;border:1px solid #bbb;background:#f3f3f3;cursor:pointer;font-size:14px}
.tabs button.on{background:#1f4e79;color:#fff}.bar{margin:6px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
h3{margin:18px 0 4px;border-bottom:2px solid #1f4e79;color:#1f4e79}.row{display:flex;gap:8px}.row>div{flex:1;min-width:0}
.plot{height:340px}.tall{height:460px}table{border-collapse:collapse;font-size:12px}
td,th{border:1px solid #ddd;padding:3px 8px;text-align:right}th{background:#eef2f7}td:first-child,th:first-child{text-align:left}
tr.c:hover{background:#ffffcc;cursor:pointer}.muted{color:#777}.pos{color:#c0392b}.neg{color:#1a7f37}
input[type=number]{width:62px}textarea{width:340px;height:50px}
</style></head><body>
<div class="tabs" id="tabs"></div><div id="content"></div>
<script>
const D=__DATA__, names=Object.keys(D), built={};
const $=id=>document.getElementById(id);
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const std=a=>{const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1))};
const pct=(a,x)=>100*a.filter(v=>v<=x).length/a.length;
function hl(a){const x=a.slice(0,-1),y=a.slice(1),mx=mean(x),my=mean(y);let sxy=0,sxx=0;
 for(let i=0;i<x.length;i++){sxy+=(x[i]-mx)*(y[i]-my);sxx+=(x[i]-mx)**2}
 const b=sxy/sxx;return b>0&&b<1?-Math.log(2)/Math.log(b):NaN}
const lay=(t,e={})=>Object.assign({title:{text:t,font:{size:13}},margin:{l:50,r:15,t:35,b:40},legend:{orientation:'h',y:-0.2},hovermode:'x unified'},e);
const P=(el,tr,l)=>Plotly.newPlot(el,tr,l,{responsive:true,displaylogo:false});
const table=(h,rows,cls)=>'<table><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+rows.map(r=>'<tr '+(r.attr||'')+(r.attr?' class="c"':'')+'>'+r.cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</table>';
const zc=z=>`<span class="${z>0?'pos':'neg'}">${z.toFixed(2)}</span>`;

names.forEach((n,i)=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>show(i);$('tabs').appendChild(b);
 const d=document.createElement('div');d.id='tab'+i;d.style.display='none';$('content').appendChild(d)});
function show(i){names.forEach((_,j)=>{$('tab'+j).style.display=i==j?'':'none';$('tabs').children[j].className=i==j?'on':''});
 if(!built[i]){built[i]=1;build(i)}else document.querySelectorAll('#tab'+i+' .js-plotly-plot').forEach(p=>Plotly.Plots.resize(p))}

function build(ti){
 const name=names[ti],d=D[name],T=d.tenors,N=d.dates.length,U=d.unit,K=d.L[0].length,m=T.length;
 const C=T.map((_,i)=>d.data[i]),CM=Object.fromEntries(T.map((t,i)=>[t,C[i]])),I=s=>s+'_'+ti,g=s=>$(I(s));
 const opts=T.map((t,i)=>`<option value="${i}">${t}</option>`).join('');
 const lb=()=>Math.max(20,Math.min(N-1,+g('lb').value||250));
 let lastRes=[];
 $('tab'+ti).innerHTML=`
 <div class="bar"><b>Lookback (days):</b><input id="${I('lb')}" type="number" value="250" style="width:70px">
  <span class="muted">${N} obs ${d.dates[0]} to ${d.dates[N-1]} | PCA var explained: ${d.var.map((v,k)=>'PC'+(k+1)+' '+(100*v).toFixed(1)+'%').join(', ')}</span></div>
 <h3>Snapshot</h3><div class="row"><div id="${I('snap')}" class="plot"></div><div id="${I('fwd')}" class="plot"></div></div><div id="${I('lvl')}"></div>
 <h3>Curve / fly scanner (largest |z| over lookback)</h3>
 <div class="bar">Type <select id="${I('type')}"><option value="fly">Fly (2*belly - wings)</option><option value="curve">Curve (long - short)</option></select>
  Show top <input id="${I('top')}" type="number" value="15"> <span class="muted">units: bp | click a row to load it into the PCA chart below</span></div><div id="${I('tbl')}"></div>
 <h3>PCA-adjusted curve / fly</h3>
 <div class="bar">Type <select id="${I('pt')}"><option value="fly">Fly</option><option value="curve">Curve</option></select>
  Legs <select id="${I('pa')}">${opts}</select><select id="${I('pb')}">${opts}</select><span id="${I('lc')}"><select id="${I('pc')}">${opts}</select></span>
  | Remove: ${[0,1,2].filter(k=>k<K).map(k=>`<label><input type="checkbox" id="${I('k'+k)}" ${k<2?'checked':''}>PC${k+1}</label>`).join(' ')}
  | <label><input type="checkbox" id="${I('auto')}" checked>Auto PC-neutral weights</label>
  Weights <input id="${I('wa')}" type="number" step="0.01"><input id="${I('wb')}" type="number" step="0.01"><span id="${I('wcs')}"><input id="${I('wc')}" type="number" step="0.01"></span></div>
 <div id="${I('pp')}" class="plot tall"></div><div id="${I('pi')}"></div>
 <h3>Plotter</h3>
 <div class="bar"><select id="${I('ps')}" multiple size="6">${opts}</select>
  <div>Custom expressions, one per line, use [tenor], e.g. 2*[5Y]-[2Y]-[10Y]<br><textarea id="${I('pe')}"></textarea></div>
  <div>Transform <select id="${I('ptr')}"><option value="lvl">Level</option><option value="chg">n-day change (bp)</option><option value="z">Rolling z (lookback)</option></select>
  n <input id="${I('pn')}" type="number" value="1"><br><button id="${I('pb2')}">Plot</button></div></div><div id="${I('pl')}" class="plot tall"></div>
 <h3>PCA diagnostics</h3>
 <div class="bar">PCs removed for residual RV <select id="${I('rn')}">${[1,2,3].filter(k=>k<=K).map(k=>`<option ${k==K?'selected':''}>${k}</option>`).join('')}</select></div>
 <div class="row"><div id="${I('ld')}" class="plot"></div><div id="${I('sc')}" class="plot"></div><div id="${I('rs')}" class="plot"></div></div>`;

 // ---- snapshot ----
 function snap(){const L=lb(),lags=[0,5,21,63].filter(l=>l<N);
  P(g('snap'),lags.map(l=>({x:T,y:C.map(c=>c[N-1-l]),name:l?l+'d ago':'today',mode:'lines+markers'})),lay('Curve snapshot',{hovermode:'closest'}));
  if(d.zero){const f=l=>T.slice(1).map((t,i)=>{const r1=C[i][N-1-l]/100,r2=C[i+1][N-1-l]/100,a=d.tn[i],b=d.tn[i+1];return 100*(Math.pow(Math.pow(1+r2,b)/Math.pow(1+r1,a),1/(b-a))-1)});
   P(g('fwd'),[0,21,63].filter(l=>l<N).map(l=>({x:T.slice(1).map((t,i)=>T[i]+'→'+t),y:f(l),name:l?l+'d ago':'today',mode:'lines+markers'})),lay('Implied forwards between adjacent tenors (annual comp.)',{hovermode:'closest'}))}
  else g('fwd').style.display='none';
  g('lvl').innerHTML=table(['Tenor','Level','1d (bp)','1w (bp)','1m (bp)','z (lb)','pctile (lb)'],T.map((t,i)=>{
   const c=C[i],cur=c[N-1],s=c.slice(-L),z=(cur-mean(s))/std(s),ch=k=>((cur-c[N-1-k])*U).toFixed(1);
   return {cells:[t,cur.toFixed(3),ch(1),ch(5),ch(21),zc(z),pct(s,cur).toFixed(0)]}}))}

 // ---- scanner ----
 function combos(type){const o=[];
  if(type==='fly'){for(let i=0;i<m;i++)for(let j=i+1;j<m;j++)for(let k=j+1;k<m;k++)o.push({ix:[i,j,k],w:[-1,2,-1]})}
  else for(let i=0;i<m;i++)for(let j=i+1;j<m;j++)o.push({ix:[i,j],w:[-1,1]});return o}
 function scan(){const L=lb(),type=g('type').value,res=[];
  for(const c of combos(type)){const s=new Array(L);
   for(let t=0;t<L;t++){let v=0;const tt=N-L+t;for(let q=0;q<c.ix.length;q++)v+=c.w[q]*C[c.ix[q]][tt];s[t]=v*U}
   const cur=s[L-1],mu=mean(s),sd=std(s);if(!(sd>0))continue;
   res.push({c,cur,mu,sd,z:(cur-mu)/sd,p:pct(s,cur),d1:cur-s[L-2],d5:cur-s[Math.max(0,L-6)],hl:hl(s)})}
  res.sort((a,b)=>Math.abs(b.z)-Math.abs(a.z));lastRes=res.slice(0,+g('top').value||15);
  g('tbl').innerHTML=table(['Trade','Current','Mean','Std','z','pctile','1d','5d','Half-life (d)'],lastRes.map((r,i)=>({attr:`data-i="${i}"`,
   cells:[r.c.ix.map(q=>T[q]).join('/'),r.cur.toFixed(2),r.mu.toFixed(2),r.sd.toFixed(2),zc(r.z),r.p.toFixed(0),r.d1.toFixed(2),r.d5.toFixed(2),isNaN(r.hl)?'n/a':r.hl.toFixed(0)]})));
  g('tbl').querySelectorAll('tr.c').forEach(tr=>tr.onclick=()=>{const r=lastRes[+tr.dataset.i],ix=r.c.ix;
   g('pt').value=type;g('pa').value=ix[0];g('pb').value=ix[1];if(ix[2]!==undefined)g('pc').value=ix[2];pcaDraw();g('pp').scrollIntoView({behavior:'smooth'})})}

 // ---- PCA fly ----
 function autoW(ix,tk){const Lm=d.L;
  if(ix.length===2){if(!tk.length)return[-1,1];const k=tk[0];return[-Lm[ix[1]][k]/Lm[ix[0]][k],1]}
  if(!tk.length)return[-1,2,-1];const[a,b,c]=ix;
  if(tk.length===1){const k=tk[0],s=-2*Lm[b][k]/(Lm[a][k]+Lm[c][k]);return[s,2,s]}
  let Saa=0,Sac=0,Scc=0,Sab=0,Scb=0;
  tk.forEach(k=>{const A=Lm[a][k],B=Lm[b][k],Cc=Lm[c][k];Saa+=A*A;Sac+=A*Cc;Scc+=Cc*Cc;Sab+=A*B;Scb+=Cc*B});
  const det=Saa*Scc-Sac*Sac;return[(-2*Sab*Scc+2*Scb*Sac)/det,2,(-2*Scb*Saa+2*Sab*Sac)/det]}
 function pcaDraw(){const fly=g('pt').value==='fly',L=lb();
  const ix=[+g('pa').value,+g('pb').value].concat(fly?[+g('pc').value]:[]);
  g('lc').style.display=g('wcs').style.display=fly?'':'none';
  const tk=[0,1,2].filter(k=>k<K&&g('k'+k).checked),raw=fly?[-1,2,-1]:[-1,1];
  let w;if(g('auto').checked){w=autoW(ix,tk);['wa','wb','wc'].forEach((s,i)=>g(s).value=i<w.length?w[i].toFixed(3):'')}
  else w=['wa','wb','wc'].slice(0,ix.length).map(s=>+g(s).value||0);
  const comb=wv=>Array.from({length:N},(_,t)=>{let v=0;ix.forEach((q,j)=>v+=wv[j]*C[q][t]);return v*U});
  const expo=(wv,k)=>ix.reduce((s,q,j)=>s+wv[j]*d.L[q][k],0);
  const proj=wv=>{const s=comb(wv);tk.forEach(k=>{const e=expo(wv,k)*U;for(let t=0;t<N;t++)s[t]-=e*d.S[t][k]});return s};
  const lab=ix.map(q=>T[q]).join('/'),S={['Raw '+lab]:comb(raw),['Raw, PCs ['+tk.map(k=>k+1)+'] projected out']:proj(raw),['Chosen weights ('+w.map(x=>x.toFixed(2))+')']:comb(w)};
  const keys=Object.keys(S),ch=S[keys[2]].slice(-L),mu=mean(ch),sd=std(ch);
  const shapes=[[mu,'#555','solid'],[mu+sd,'#999','dot'],[mu-sd,'#999','dot'],[mu+2*sd,'#bbb','dash'],[mu-2*sd,'#bbb','dash']].map(([y,c,dash])=>({type:'line',xref:'paper',x0:0,x1:1,y0:y,y1:y,line:{color:c,dash,width:1}}));
  P(g('pp'),keys.map((k,i)=>({x:d.dates,y:S[k],name:k,mode:'lines',line:{width:i==2?2:1.2},visible:i==1?'legendonly':true})),
   lay('Chosen-weights series with lookback mean and ±1/±2σ bands (bp)',{shapes,xaxis:{rangeslider:{visible:true}},height:460}));
  const hdr=['Series','Weights','Current (bp)','z (lb)',...Array.from({length:K},(_,k)=>'PC'+(k+1)+' exposure')];
  const row=(n,wv,s,ex)=>{const x=s.slice(-L),c=x[L-1];return{cells:[n,wv.map(v=>v.toFixed(2)).join(' / '),c.toFixed(2),zc((c-mean(x))/std(x)),...Array.from({length:K},(_,k)=>ex?expo(wv,k).toFixed(3):'removed')]}};
  g('pi').innerHTML=table(hdr,[row(keys[0],raw,S[keys[0]],1),row(keys[1],raw,S[keys[1]],0),row(keys[2],w,S[keys[2]],1)])+
   '<div class="muted">Exposure = Σ weight × PC loading (change in series per 1-unit move in that PC, in data units). Auto weights fix belly=+2 (fly) / long leg=+1 (curve) and solve the other legs; with 3 PCs on a fly it is a least-squares fit; a 2-leg curve can only neutralise the first ticked PC.</div>'}
 ['type','top','lb'].forEach(s=>g(s).onchange=()=>{if(s=='lb'){snap();diag();pcaDraw()}scan()});
 ['pt','pa','pb','pc','auto','wa','wb','wc'].concat([0,1,2].filter(k=>k<K).map(k=>'k'+k)).forEach(s=>g(s).onchange=pcaDraw);

 // ---- plotter ----
 function plot(){const tr=g('ptr').value,n=+g('pn').value||1,L=lb(),ser=[];
  [...g('ps').selectedOptions].forEach(o=>ser.push([T[o.value],C[o.value]]));
  g('pe').value.split('\n').map(s=>s.trim()).filter(Boolean).forEach(e=>{try{
   const f=new Function('c','t','return '+e.replace(/\[([^\]]+)\]/g,'c["$1"][t]'));ser.push([e,Array.from({length:N},(_,t)=>f(CM,t))])}catch(err){alert('Bad expression: '+e)}});
  const out=ser.map(([nm,y])=>{let v=y;
   if(tr==='chg')v=y.map((x,t)=>t>=n?(x-y[t-n])*U:null);
   if(tr==='z')v=y.map((x,t)=>{if(t<L)return null;const w=y.slice(t-L+1,t+1);return(x-mean(w))/std(w)});
   return{x:d.dates,y:v,name:nm,mode:'lines'}});
  P(g('pl'),out,lay('Plotter',{xaxis:{rangeslider:{visible:true}}}))}
 g('pb2').onclick=plot;

 // ---- PCA diagnostics ----
 function diag(){const L=lb(),np=+g('rn').value;
  P(g('ld'),Array.from({length:K},(_,k)=>({x:T,y:d.L.map(r=>r[k]),name:`PC${k+1} (${(100*d.var[k]).toFixed(1)}%)`,mode:'lines+markers'})),lay('PC loadings',{hovermode:'closest'}));
  P(g('sc'),Array.from({length:K},(_,k)=>({x:d.dates,y:d.S.map(r=>r[k]*U),name:'PC'+(k+1)})),lay('PC scores (bp)'));
  const z=[],cur=[];T.forEach((t,i)=>{const r=C[i].map((x,tt)=>{let v=x-d.mean[i];for(let k=0;k<np;k++)v-=d.L[i][k]*d.S[tt][k];return v*U}).slice(-L);
   cur.push(r[L-1]);z.push((r[L-1]-mean(r))/std(r))});
  P(g('rs'),[{type:'bar',x:T,y:cur,text:z.map(v=>'z='+v.toFixed(2)),marker:{color:z.map(v=>v>0?'#c0392b':'#1a7f37')}}],lay('Residual vs '+np+'-PC fit today (bp, + = high yield/cheap)',{hovermode:'closest'}))}

 const mid=Math.floor(m/2);g('pa').value=0;g('pb').value=mid;g('pc').value=m-1;
 snap();scan();pcaDraw();diag();
 g('rn').onchange=diag;
}
show(0);
</script></body></html>"""

def build_dashboard(outfile=OUTFILE):
    data = {name: prep(c["df"], c["unit"], c["zero"]) for name, c in CONFIG.items()}
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(outfile, "w", encoding="utf-8") as f: f.write(html)
    return outfile

_f = build_dashboard()
display(IFrame(_f, width="100%", height=1100))
