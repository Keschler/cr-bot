const COLORS = ['#75bcff', '#f1878d'];
const fallbackGeometry = {width:18,height:32,river:{y_min:15,y_max:17},bridges:[{x_min:3,x_max:4,y_min:15,y_max:17},{x_min:14,x_max:15,y_min:15,y_max:17}]};
const niceName = value => String(value ?? 'Unknown').replaceAll('_',' ').replaceAll('-',' ');

/** Canvas coordinates use authoritative simulator tiles: x right, y down. */
export class Arena {
  constructor(canvas, onSelect) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSelect = onSelect;
    this.geometry = fallbackGeometry;
    this.frame = null;
    this.selected = null;
    this.options = {grid:false,labels:true,targets:true,ranges:false,heatmap:false};
    this.images = new Map();
    this.transform = {x:0,y:0,scale:1};
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    canvas.addEventListener('click', event => this.pick(event));
    canvas.addEventListener('keydown', event => {if(event.key === 'Escape') this.select(null);});
  }
  setGeometry(geometry) {this.geometry = {...fallbackGeometry,...geometry};this.draw();}
  setFrame(frame) {
    this.frame = frame;
    if (this.selected != null && !frame?.entities?.some(entity => String(entity.uid) === String(this.selected))) this.selected = null;
    this.draw();
  }
  setOption(name,value) {this.options[name]=value;this.draw();}
  select(uid) {this.selected=uid;this.draw();this.onSelect?.(uid);}
  pick(event) {
    const rect = this.canvas.getBoundingClientRect(), t=this.transform;
    const x = (event.clientX-rect.left-t.x)/t.scale, y=(event.clientY-rect.top-t.y)/t.scale;
    const found = (this.frame?.entities??[]).filter(e => Number.isFinite(e.x)&&Number.isFinite(e.y)).map(entity => ({entity,distance:Math.hypot(entity.x-x,entity.y-y)})).filter(({entity,distance})=>distance<Math.max(.7,Number(entity.radius)||.5)).sort((a,b)=>a.distance-b.distance)[0];
    this.select(found?.entity.uid??null);
  }
  image(card) {
    if(!card)return null;
    if(!this.images.has(card)) {
      const img=new Image();
      img.onload=()=>this.draw();
      img.onerror=()=>{this.images.set(card,null);this.draw();};
      this.images.set(card,img);
      img.src=`/api/card-icon?name=${encodeURIComponent(card)}`;
    }
    const img=this.images.get(card);
    return img?.complete && img.naturalWidth ? img:null;
  }
  draw() {
    const canvas=this.canvas, ctx=this.ctx;
    const box=canvas.getBoundingClientRect();
    if(!box.width||!box.height)return;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    if(canvas.width!==Math.round(box.width*dpr)||canvas.height!==Math.round(box.height*dpr)) {
      canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
    }
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,box.width,box.height);
    const g=this.geometry,w=g.width,h=g.height;
    const scale=Math.min((box.width-34)/w,(box.height-42)/h), x=(box.width-w*scale)/2,y=(box.height-h*scale)/2;
    this.transform={x,y,scale};
    ctx.fillStyle='#102024';ctx.fillRect(0,0,box.width,box.height);
    // Quiet coordinate marks preserve the feel of a tactical board at any size.
    ctx.strokeStyle='#284048';ctx.lineWidth=1;
    for(let iy=0;iy<box.height;iy+=24){ctx.beginPath();ctx.moveTo(0,iy+.5);ctx.lineTo(box.width,iy+.5);ctx.stroke();}
    const vignette=ctx.createLinearGradient(0,0,box.width,0);vignette.addColorStop(0,'#0c1b20aa');vignette.addColorStop(.5,'#0c1b2000');vignette.addColorStop(1,'#0c1b20aa');ctx.fillStyle=vignette;ctx.fillRect(0,0,box.width,box.height);
    ctx.save();ctx.translate(x,y);ctx.scale(scale,scale);
    ctx.fillStyle='#1c3534';ctx.fillRect(0,0,w,h);
    for(let row=0;row<h;row++)for(let col=0;col<w;col++) {
      ctx.fillStyle=(row+col)%2?'#23403a':'#213d37';ctx.fillRect(col,row,1,1);
    }
    ctx.fillStyle='#61726110';ctx.fillRect(2,0,2.7,h);ctx.fillRect(w-4.7,0,2.7,h);
    ctx.strokeStyle='#63856850';ctx.lineWidth=.045;ctx.strokeRect(.06,.06,w-.12,h-.12);
    ctx.setLineDash([.3,.3]);ctx.strokeStyle='#739b7622';ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke();ctx.setLineDash([]);
    const river=g.river??fallbackGeometry.river, ry=river.y_min??river.top??15,rh=(river.y_max??river.bottom??17)-ry;
    ctx.fillStyle='#214957';ctx.fillRect(0,ry,w,rh);ctx.fillStyle='#3f758055';ctx.fillRect(0,ry,w,.12);ctx.fillRect(0,ry+rh-.12,w,.12);
    ctx.strokeStyle='#65a3a922';ctx.lineWidth=.035;
    for(let iy=ry+.3;iy<ry+rh;iy+=.35){ctx.beginPath();ctx.moveTo(0,iy);ctx.lineTo(w,iy);ctx.stroke();}
    for(const bridge of g.bridges??fallbackGeometry.bridges){
      const bx=bridge.x_min,by=bridge.y_min??ry,bw=bridge.x_max-bx,bh=(bridge.y_max??ry+rh)-by;
      ctx.fillStyle='#77715b';ctx.fillRect(bx-.12,by-.12,bw+.24,bh+.24);
      ctx.fillStyle='#9a8a67';ctx.fillRect(bx,by,bw,bh);
      ctx.strokeStyle='#4f5148';ctx.lineWidth=.065;
      for(let iy=by+.2;iy<by+bh;iy+=.35){ctx.beginPath();ctx.moveTo(bx,iy);ctx.lineTo(bx+bw,iy);ctx.stroke();}
    }
    this.drawHeatmap(ctx,w,h);
    if(this.options.grid){ctx.strokeStyle='#9bbbab22';ctx.lineWidth=.035;ctx.beginPath();for(let ix=0;ix<=w;ix++){ctx.moveTo(ix,0);ctx.lineTo(ix,h);}for(let iy=0;iy<=h;iy++){ctx.moveTo(0,iy);ctx.lineTo(w,iy);}ctx.stroke();}
    if(this.frame) {
      const entities=this.frame.entities??[],byId=new Map(entities.map(e=>[String(e.uid),e]));
      for(const effect of this.frame.effects??[])this.drawEffect(ctx,effect);
      for(const entity of entities){
        if(!Number.isFinite(entity.x)||!Number.isFinite(entity.y))continue;
        const selected=String(entity.uid)===String(this.selected);
        if((this.options.ranges||selected)&&Number(entity.range)>0){ctx.beginPath();ctx.arc(entity.x,entity.y,entity.range,0,Math.PI*2);ctx.fillStyle=selected?'#aad77509':'#6b8fcd04';ctx.fill();ctx.strokeStyle=selected?'#b7f46b99':`${COLORS[entity.owner]??COLORS[0]}35`;ctx.lineWidth=selected?.07:.035;ctx.setLineDash([.2,.18]);ctx.stroke();ctx.setLineDash([]);}
        if((this.options.targets||selected)&&entity.target_uid!=null){const target=byId.get(String(entity.target_uid));if(target)this.arrow(ctx,entity.x,entity.y,target.x,target.y,`${COLORS[entity.owner]??'#c2d0df'}70`,.045);}
      }
      for(const projectile of this.frame.projectiles??[])this.drawProjectile(ctx,projectile);
      const sorted=[...entities].sort((a,b)=>(a.kind==='tower'?-1:0)-(b.kind==='tower'?-1:0));
      for(const entity of sorted)this.drawEntity(ctx,entity,scale);
    }
    ctx.restore();
    ctx.font='8px ui-monospace, monospace';ctx.fillStyle='#66817e';ctx.textAlign='center';
    if(box.width-w*scale>42){for(let row=0;row<h;row+=4){ctx.fillText(String(row).padStart(2,'0'),x-12,y+row*scale+3);ctx.fillText(String(row).padStart(2,'0'),x+w*scale+12,y+row*scale+3);}}
    for(let col=0;col<w;col+=3)ctx.fillText(String(col).padStart(2,'0'),x+(col+.5)*scale,y+h*scale+12);
  }
  drawHeatmap(ctx,w,h) {
    const heatmap=this.frame?.policy?.heatmap;
    if(!this.options.heatmap||!Array.isArray(heatmap)||!heatmap.length)return;
    const max=Math.max(...heatmap.filter(Number.isFinite));
    if(!(max>0))return;
    for(let i=0;i<Math.min(heatmap.length,w*h);i++){
      const strength=Math.pow(Math.max(0,heatmap[i]||0)/max,.6);
      if(strength<.02)continue;
      ctx.fillStyle=`rgba(199,239,92,${strength*.62})`;ctx.fillRect(i%w,Math.floor(i/w),1,1);
    }
  }
  drawEffect(ctx,effect) {
    if(!Number.isFinite(effect.x)||!Number.isFinite(effect.y))return;
    const color=COLORS[effect.owner]??'#e2bf82';ctx.beginPath();ctx.arc(effect.x,effect.y,Math.max(.3,effect.radius??1),0,Math.PI*2);ctx.fillStyle=`${color}1c`;ctx.fill();ctx.strokeStyle=`${color}88`;ctx.lineWidth=.08;ctx.setLineDash([.18,.16]);ctx.stroke();ctx.setLineDash([]);
  }
  drawProjectile(ctx,p) {
    if(!Number.isFinite(p.x)||!Number.isFinite(p.y))return;
    const color=COLORS[p.owner]??'#ecdb9c';
    if(Number.isFinite(p.target_x)&&Number.isFinite(p.target_y)){const dx=p.target_x-p.x,dy=p.target_y-p.y,d=Math.hypot(dx,dy)||1;ctx.beginPath();ctx.moveTo(p.x-dx/d*.7,p.y-dy/d*.7);ctx.lineTo(p.x,p.y);ctx.strokeStyle=`${color}80`;ctx.lineWidth=.09;ctx.stroke();}
    ctx.beginPath();ctx.arc(p.x,p.y,Math.max(.09,Math.min(p.radius??.13,.3)),0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
  }
  arrow(ctx,x1,y1,x2,y2,color,width) {
    const angle=Math.atan2(y2-y1,x2-x1),offset=.55,tx=x2-Math.cos(angle)*offset,ty=y2-Math.sin(angle)*offset;
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(tx,ty);ctx.stroke();ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-Math.cos(angle-.5)*.35,ty-Math.sin(angle-.5)*.35);ctx.lineTo(tx-Math.cos(angle+.5)*.35,ty-Math.sin(angle+.5)*.35);ctx.closePath();ctx.fill();
  }
  drawEntity(ctx,entity,scale) {
    if(!Number.isFinite(entity.x)||!Number.isFinite(entity.y))return;
    const {x,y}=entity,owner=Number(entity.owner),color=COLORS[owner]??'#c4bca5',selected=String(entity.uid)===String(this.selected);
    const tower=String(entity.kind).includes('tower')||String(entity.role).includes('tower')||/tower/i.test(entity.card_id??'');
    const radius=tower?Math.max(.65,Math.min(1.3,Number(entity.radius)||.9)):Math.max(.35,Math.min(.67,Number(entity.radius)||.45));
    ctx.save();ctx.globalAlpha=entity.deploying?.58:1;
    if(selected){ctx.beginPath();ctx.arc(x,y,radius+.2,0,Math.PI*2);ctx.strokeStyle='#c5f38c';ctx.lineWidth=.09;ctx.stroke();}
    ctx.beginPath();if(tower)ctx.roundRect(x-radius,y-radius,radius*2,radius*2,.16);else ctx.arc(x,y,radius,0,Math.PI*2);
    ctx.fillStyle=owner===1?'#57383b':'#2c4b65';ctx.fill();
    const img=this.image(entity.card_id);
    if(img){ctx.save();ctx.clip();const crop=Math.min(img.naturalWidth,img.naturalHeight);ctx.drawImage(img,(img.naturalWidth-crop)/2,(img.naturalHeight-crop)/2,crop,crop,x-radius,y-radius,radius*2,radius*2);ctx.restore();}
    else {ctx.fillStyle=color;ctx.font=`600 ${tower?.6:.4}px ui-monospace,monospace`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(tower?'♜':niceName(entity.card_id).slice(0,2).toUpperCase(),x,y);}
    ctx.strokeStyle=color;ctx.lineWidth=tower?.11:.085;ctx.stroke();
    if(Number(entity.max_hp)>0){const ratio=Math.max(0,Math.min(1,entity.hp/entity.max_hp)),bw=radius*2+.13,by=y-radius-.29;ctx.fillStyle='#111d28';ctx.fillRect(x-bw/2,by,bw,.14);ctx.fillStyle=ratio<.25?'#fcb776':color;ctx.fillRect(x-bw/2,by,bw*ratio,.14);}
    if(this.options.labels||selected){const label=niceName(entity.card_id);ctx.font=`${Math.max(7,Math.min(9,scale*.55))/scale}px ui-sans-serif,sans-serif`;ctx.textAlign='center';ctx.textBaseline='top';ctx.strokeStyle='#102025';ctx.lineWidth=.18;ctx.strokeText(label,x,y+radius+.13);ctx.fillStyle='#d6e3df';ctx.fillText(label,x,y+radius+.13);}
    if(entity.statuses?.length){ctx.beginPath();ctx.arc(x+radius*.7,y+radius*.6,.14,0,Math.PI*2);ctx.fillStyle='#edd482';ctx.fill();}
    ctx.restore();
  }
}
