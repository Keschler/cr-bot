export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
export const titleCase = value => String(value).replaceAll('_',' ').replace(/\b\w/g,char=>char.toUpperCase());
export function formatMetric(value,key='') {
  if(typeof value!=='number'||!Number.isFinite(value))return '—';
  if(/win_rate|accuracy|fraction|rate$/.test(key)&&Math.abs(value)<=1)return `${(value*100).toFixed(1)}%`;
  if(Math.abs(value)>=1000000)return `${(value/1000000).toFixed(2)}M`;
  if(Math.abs(value)>=10000)return `${(value/1000).toFixed(1)}k`;
  if(value!==0&&Math.abs(value)<.0001)return value.toExponential(2);
  if(Number.isInteger(value))return value.toLocaleString();
  return value.toFixed(Math.abs(value)<.01?5:Math.abs(value)<1?3:2);
}
export function metricKeys(runs) {
  const keys=new Set();
  for(const run of runs)for(const point of run?.series??[])for(const [key,value] of Object.entries(point))if(key!=='step'&&typeof value==='number'&&Number.isFinite(value))keys.add(key);
  const order=['win_rate','reward','policy_loss','value_loss','entropy','kl'];
  return [...keys].sort((a,b)=>(order.includes(a)?order.indexOf(a):100)-(order.includes(b)?order.indexOf(b):100)||a.localeCompare(b));
}
export function latestMetric(run,key){return [...(run?.series??[])].reverse().find(point=>typeof point[key]==='number'&&Number.isFinite(point[key]))?.[key];}

class LineChart {
  constructor(canvas,runs,key,readout) {
    this.canvas=canvas;this.runs=runs;this.key=key;this.readout=readout;this.hover=null;
    this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(canvas);
    canvas.addEventListener('pointermove',event=>{this.hover=event.offsetX;this.draw();});
    canvas.addEventListener('pointerleave',()=>{this.hover=null;this.readout.textContent='BY TRAINING STEP';this.draw();});
    this.draw();
  }
  destroy(){this.observer.disconnect();}
  draw(){
    const canvas=this.canvas,box=canvas.getBoundingClientRect();if(!box.width||!box.height)return;
    const dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const p={left:53,right:17,top:12,bottom:29},w=box.width-p.left-p.right,h=box.height-p.top-p.bottom;
    const valid=this.runs.flatMap(run=>(run.series??[]).filter(point=>Number.isFinite(point.step)&&Number.isFinite(point[this.key])));
    if(!valid.length)return;
    let xMin=Math.min(...valid.map(d=>d.step)),xMax=Math.max(...valid.map(d=>d.step));if(xMin===xMax){xMin=Math.min(0,xMin);xMax=Math.max(xMin+1,xMax);}
    let yMin=Math.min(...valid.map(d=>d[this.key])),yMax=Math.max(...valid.map(d=>d[this.key]));const span=yMax-yMin||Math.abs(yMax)*.2||1;yMin-=span*.12;yMax+=span*.12;
    if(/win_rate/.test(this.key)){yMin=Math.max(0,yMin);yMax=Math.min(1,yMax);if(yMin===yMax)yMax=yMin+1;}
    const px=x=>p.left+(x-xMin)/(xMax-xMin)*w,py=y=>p.top+h-(y-yMin)/(yMax-yMin)*h;
    ctx.font='9px ui-monospace,monospace';ctx.lineWidth=1;
    for(let i=0;i<5;i++){const value=yMin+(yMax-yMin)*i/4,y=py(value);ctx.strokeStyle='#29405a70';ctx.beginPath();ctx.moveTo(p.left,y+.5);ctx.lineTo(p.left+w,y+.5);ctx.stroke();ctx.fillStyle='#69829d';ctx.textAlign='right';ctx.fillText(formatMetric(value,this.key),p.left-9,y+3);}
    ctx.textAlign='center';for(let i=0;i<4;i++){const value=xMin+(xMax-xMin)*i/3;ctx.fillStyle='#69829d';ctx.fillText(formatMetric(value,'step'),px(value),p.top+h+19);}
    const colors=['#70b9fb','#f4c36d'];
    this.runs.forEach((run,runIndex)=>{
      const points=(run.series??[]).filter(point=>Number.isFinite(point.step)).sort((a,b)=>a.step-b.step),color=colors[runIndex%2];
      ctx.save();ctx.beginPath();ctx.rect(p.left-3,p.top-3,w+6,h+6);ctx.clip();
      ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.lineJoin='round';ctx.beginPath();let connected=false;
      for(const point of points){if(!Number.isFinite(point[this.key])){connected=false;continue;}const x=px(point.step),y=py(point[this.key]);if(connected)ctx.lineTo(x,y);else ctx.moveTo(x,y);connected=true;}
      ctx.stroke();
      // Dots also make isolated measurements and intentional gaps visible.
      if(valid.length<80)for(const point of points){if(!Number.isFinite(point[this.key]))continue;ctx.beginPath();ctx.arc(px(point.step),py(point[this.key]),2,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
      ctx.restore();
    });
    if(this.hover!=null&&this.hover>=p.left&&this.hover<=p.left+w){const step=xMin+(this.hover-p.left)/w*(xMax-xMin);ctx.strokeStyle='#b4c9dd55';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(this.hover,p.top);ctx.lineTo(this.hover,p.top+h);ctx.stroke();ctx.setLineDash([]);const labels=[];
      this.runs.forEach((run,i)=>{const points=(run.series??[]).filter(point=>Number.isFinite(point.step)&&Number.isFinite(point[this.key]));if(!points.length)return;const nearest=points.reduce((a,b)=>Math.abs(a.step-step)<Math.abs(b.step-step)?a:b);ctx.fillStyle=colors[i%2];ctx.beginPath();ctx.arc(px(nearest.step),py(nearest[this.key]),3,0,Math.PI*2);ctx.fill();labels.push(`${this.runs.length>1?String.fromCharCode(65+i)+': ':''}${formatMetric(nearest[this.key],this.key)}`);});this.readout.textContent=`STEP ${formatMetric(step,'step')} · ${labels.join(' / ')}`;
    }
  }
}

export function renderRunAnalytics(container,runs,onCheckpoint){
  const keys=metricKeys(runs),compare=runs.length===2,metrics=['win_rate','reward','entropy','step'];
  const latestStep=run=>{const steps=(run.series??[]).map(p=>p.step).filter(Number.isFinite);return steps.length?Math.max(...steps):null;};
  const summaries=metrics.map(key=>{
    const values=runs.map(run=>key==='step'?latestStep(run):latestMetric(run,key));
    const delta=values.length===2&&values.every(Number.isFinite)?values[1]-values[0]:null;
    const detail=compare?`<small><span class="compare-name">A ${formatMetric(values[0],key)}</span> · <span class="compare-name alt">B ${formatMetric(values[1],key)}</span></small>`:`<small>${key==='step'?`${runs[0].series?.length??0} recorded measurements`:'Latest recorded value'}</small>`;
    return `<div class="summary-card"><span>${key==='step'?'Training step':key==='reward'?'Mean return':titleCase(key)}${compare?' · B − A':''}</span><strong>${compare?(delta==null?'—':`${delta>0?'+':''}${formatMetric(delta,key)}`):formatMetric(values[0],key)}</strong>${detail}</div>`;
  }).join('');
  container.className='analytics-results';
  container.innerHTML=`<div class="summary-cards">${summaries}</div><div class="chart-grid">${keys.map(key=>`<section class="chart-card"><div class="chart-heading"><h3>${escapeHTML(key==='reward'?'Mean return':titleCase(key))}</h3><span data-readout="${escapeHTML(key)}">BY TRAINING STEP</span></div><div class="chart-area"><canvas data-chart="${escapeHTML(key)}" aria-label="${escapeHTML(key==='reward'?'Mean return':titleCase(key))} over training steps"></canvas></div><div class="chart-legend">${runs.map((run,i)=>`<span><i class="dot ${i===0?'blue':'amber'}"></i>${escapeHTML(run.name)}</span>`).join('')}</div></section>`).join('')}</div>${!keys.length?'<div class="analytics-empty"><h2>No numeric training history</h2><p>This run has no compatible recorded metrics yet. Checkpoints and raw run metadata are listed below.</p></div>':''}<p class="metrics-note">Missing measurements appear as gaps. Hover over a chart to inspect the nearest recorded sample. ${compare?'Deltas compare each run’s latest available value; their training steps may differ.':''}</p>${runs.map((run,index)=>checkpointTable(run,index,compare)).join('')}<details class="raw-details"><summary>Run paths and raw metadata</summary><pre>${escapeHTML(JSON.stringify(runs.map(({path,summary,raw})=>({path,summary,raw})),null,2))}</pre></details>`;
  const charts=[];
  for(const canvas of container.querySelectorAll('[data-chart]'))charts.push(new LineChart(canvas,runs,canvas.dataset.chart,container.querySelector(`[data-readout="${CSS.escape(canvas.dataset.chart)}"]`)));
  container.querySelectorAll('[data-checkpoint]').forEach(button=>button.addEventListener('click',()=>onCheckpoint(button.dataset.checkpoint)));
  return ()=>charts.forEach(chart=>chart.destroy());
}
function checkpointTable(run,index,compare){
  const rows=(run.checkpoints??[]).map(cp=>typeof cp==='string'?{path:cp,name:cp.split('/').pop()}:cp);
  return `<div class="data-table-wrap"><div class="data-table-title">${compare?`<span class="compare-name ${index===1?'alt':''}">${index===0?'A':'B'} · </span>`:''}${escapeHTML(run.name)} <span class="muted">/ ${rows.length} checkpoints</span></div><table class="data-table"><thead><tr><th>CHECKPOINT</th><th>STEP</th><th>WIN RATE</th><th>MEAN RETURN</th><th>MODIFIED</th><th></th></tr></thead><tbody>${rows.length?rows.map(cp=>`<tr><td title="${escapeHTML(cp.path)}">${escapeHTML(cp.name??cp.path?.split('/').pop()??'Checkpoint')}</td><td>${formatMetric(cp.step??cp.update,'step')}</td><td>${formatMetric(cp.metrics?.win_rate,'win_rate')}</td><td>${formatMetric(cp.metrics?.reward,'reward')}</td><td>${escapeHTML(cp.modified_at?new Date(typeof cp.modified_at==='number'?cp.modified_at*1000:cp.modified_at).toLocaleString():'—')}</td><td>${cp.path?`<button class="text-button" data-checkpoint="${escapeHTML(cp.path)}">Watch exhibition ↗</button>`:''}</td></tr>`).join(''):'<tr><td colspan="6">No checkpoints found in this run.</td></tr>'}</tbody></table></div>`;
}
