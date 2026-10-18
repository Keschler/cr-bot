import {Arena} from './arena.js';
import {escapeHTML as esc,formatMetric,renderRunAnalytics} from './charts.js';
import {playbackCommand} from './keyboard.js';

const $=selector=>document.querySelector(selector);
const $$=selector=>[...document.querySelectorAll(selector)];
const state={catalog:{checkpoints:[],decks:[],archetypes:[]},library:{matches:[],runs:[]},source:'all',query:'',session:null,selectedPath:null,frames:[],position:0,playing:false,following:false,speed:1,clock:0,page:'watch',inspector:'overview',generation:0,polling:false,libraryLoading:false,eventLimit:300,runCache:new Map(),charts:{training:null,compare:null},runGeneration:{training:0,compare:0}};
const arena=new Arena($('#arena'),()=>renderEntity());
let noticeTimer,searchTimer,lastAnimation=0,timelineStamp="";
const cardName=card=>typeof card==='object'&&card!==null?String(card.card_id??card.name??card.card??card.id??'Unknown'):String(card??'Unknown');
const pretty=value=>String(value??'—').replaceAll('_',' ');
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const asDate=value=>typeof value==='number'?new Date(value*1000):new Date(value);
const clock=(value,precise=false)=>{const n=Math.max(0,Number(value)||0),minutes=Math.floor(n/60),seconds=n%60;return `${String(minutes).padStart(2,'0')}:${precise?seconds.toFixed(1).padStart(4,'0'):String(Math.floor(seconds)).padStart(2,'0')}`;};
const currentFrame=()=>state.frames[state.position]??null;
const playerFor=(frame,id)=>frame?.players?.find(p=>Number(p.player)===id)??frame?.players?.[id];

async function api(path,options={}){
  const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...options.headers}});
  let data;try{data=await response.json();}catch{throw new Error(`The server returned ${response.status} without a readable response.`);}
  if(!response.ok){const detail=data.detail??data.error??`Request failed (${response.status})`,error=new Error(typeof detail==='string'?detail:JSON.stringify(detail));error.status=response.status;throw error;}
  return data;
}
function notify(message,error=false){clearTimeout(noticeTimer);$('#notice').textContent=message;$('#notice').hidden=false;$('#notice').classList.toggle('error',error);noticeTimer=setTimeout(()=>{$('#notice').hidden=true;},error?14000:7000);}
function bytes(n){return n>1048576?`${(n/1048576).toFixed(1)} MB`:n>1024?`${Math.round(n/1024)} KB`:`${n??0} B`;}
function relativeDate(value){if(!value)return '';const date=asDate(value);if(!Number.isFinite(date.getTime()))return '';const age=(Date.now()-date.getTime())/1000;if(age<60)return 'just now';if(age<3600)return `${Math.floor(age/60)}m ago`;if(age<86400)return `${Math.floor(age/3600)}h ago`;return date.toLocaleDateString(undefined,{month:'short',day:'numeric'});}
function sourceLabel(source){return {training:'Training sample',frontier:'Saved replay',replay:'Saved replay',exhibition:'Exhibition'}[source]??pretty(source);}

async function loadLibrary(initial=false){
  if(state.libraryLoading)return;
  state.libraryLoading=true;
  if(initial)$('#match-list').innerHTML='<div class="loading-skeleton"></div><div class="loading-skeleton"></div><div class="loading-skeleton"></div>';
  try{
    const data=await api('/api/library?limit=500');
    state.library={matches:data.matches??[],runs:data.runs??[]};
    renderLibrary();populateRunSelectors();
    $('#library-updated').textContent='AUTO SYNC';
  }catch(error){if(initial)$('#match-list').innerHTML=`<div class="empty-library"><strong>Library unavailable</strong>${esc(error.message)}<br>Use refresh to try again.</div>`;else notify(error.message,true);}
  finally{state.libraryLoading=false;}
}
function renderLibrary(){
  const matches=state.library.matches.filter(match=>(state.source==='all'||match.source===state.source)&&(!state.query||`${match.name} ${match.path} ${match.source}`.toLowerCase().includes(state.query)));
  $('#match-count').textContent=`${matches.length} MATCH${matches.length===1?'':'ES'}`;
  const list=$('#match-list'),scrollY=list.scrollTop,scrollX=list.scrollLeft,focused=document.activeElement?.dataset?.matchPath;
  list.innerHTML=matches.length?matches.map(match=>`<button class="match-card ${match.path===state.selectedPath?'selected':''}" data-match-path="${esc(match.path)}" aria-pressed="${match.path===state.selectedPath}" title="${esc(match.path)}"><span class="match-card-top"><span class="source-mark ${esc(match.source)}"></span>${esc(sourceLabel(match.source))}${match.active?'<span class="tiny-live">● LIVE</span>':''}</span><h3>${esc(match.name??match.path?.split('/').pop())}</h3><span class="match-card-meta"><span>${esc(relativeDate(match.modified_at))}</span><span>${bytes(match.size)}</span></span></button>`).join(''):`<div class="empty-library"><strong>${state.query?'No matching matches':'No matches here yet'}</strong>${state.query?'Try a different name or source.':'Launch an exhibition, or enable capture for a new V4 training run. Saved frontier replays appear automatically.'}</div>`;
  list.scrollTop=scrollY;list.scrollLeft=scrollX;
  list.querySelectorAll('[data-match-path]').forEach(button=>button.addEventListener('click',()=>openReplay(button.dataset.matchPath)));
  if(focused)list.querySelector(`[data-match-path="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
}
async function loadCatalog(){
  try{state.catalog=await api('/api/catalog');populateCatalog();}
  catch(error){notify(`Cannot load checkpoint catalog: ${error.message}`,true);}
}
function option(value,label){return `<option value="${esc(value)}">${esc(label)}</option>`;}
function populateCatalog(){
  const checkpoints=state.catalog.checkpoints??[];
  const options=checkpoints.map(cp=>option(cp.path,cp.name??cp.path)).join('');
  for(const id of ['#launch-checkpoint','#launch-opponent-checkpoint']){const select=$(id),selected=select.value;select.innerHTML=option('','Select a checkpoint')+options;if(checkpoints.some(cp=>cp.path===selected))select.value=selected;}
  $('#launch-archetype').innerHTML=(state.catalog.archetypes??[]).map(name=>option(name,pretty(name))).join('')||option('','Default');
  const decks=state.catalog.decks??[],learnerCards=state.catalog.learner_deck??[];
  const learnerVocabulary=new Set(learnerCards.map(cardName));
  const compatible=deck=>!learnerVocabulary.size||(deck.cards?.length===8&&deck.cards.every(card=>learnerVocabulary.has(cardName(card))));
  $('#launch-learner-deck').innerHTML=option('','Checkpoint / default deck')+decks.filter(compatible).map(deck=>option(deck.id,`${deck.name??deck.id}${deck.archetype?' · '+pretty(deck.archetype):''}`)).join('');
  $('#launch-opponent-deck').innerHTML=option('','Default opponent deck')+decks.map(deck=>option(deck.id,`${deck.name??deck.id}${deck.archetype?' · '+pretty(deck.archetype):''}`)).join('');
  $('#launch-submit').disabled=!checkpoints.length;
  if(learnerCards.length)$('#learner-custom-deck').placeholder=learnerCards.map(cardName).join(', ');
  let hint=$('#deck-vocabulary-hint');if(!hint){hint=document.createElement('p');hint.id='deck-vocabulary-hint';hint.className='small muted';$('.custom-decks').after(hint);}
  hint.textContent=learnerCards.length?`V4 learner and checkpoint opponents use the checkpoint card vocabulary: ${learnerCards.map(cardName).join(', ')}. Custom learner decks may reorder these eight cards. Scripted opponents can use any supported deck.`:'Checkpoint decks must use the card vocabulary supported by the V4 policy.';
  if(!checkpoints.length){$('#launch-error').textContent='No local V4 checkpoints were found. Complete a training checkpoint, then refresh the library.';$('#launch-error').hidden=false;}
}
function populateRunSelectors(){
  for(const id of ['#training-run','#compare-run-a','#compare-run-b']){const select=$(id),selected=select.value;select.innerHTML=option('','Select a run')+state.library.runs.map(run=>option(run.id,run.name)).join('');if(state.library.runs.some(run=>run.id===selected))select.value=selected;}
}

async function openReplay(path){
  const previous=state.session;
  const generation=++state.generation;
  state.selectedPath=path;resetMatch();renderLibrary();
  $('#match-title').textContent=path.split('/').pop();$('#match-subtitle').textContent='Preparing recorded match…';$('#match-status').textContent='LOADING';$('#match-status').className='status-pill';
  try{if(previous?.state==='running')await api(`/api/matches/${encodeURIComponent(previous.id)}/stop`,{method:'POST'});if(generation!==state.generation)return;let session;for(let attempt=0;attempt<6;attempt++){try{session=await api('/api/replays',{method:'POST',body:JSON.stringify({path})});break;}catch(error){if(error.status!==409||!previous||attempt===5)throw error;await new Promise(resolve=>setTimeout(resolve,200));if(generation!==state.generation)return;}}if(generation!==state.generation)return;adoptSession(session);await pollFrames();}
  catch(error){if(generation!==state.generation)return;$('#match-subtitle').textContent=error.message;$('#match-status').textContent='ERROR';$('#match-status').classList.add('error');notify(error.message,true);}
}
function resetMatch(){
  state.session=null;state.frames=[];state.position=0;state.playing=false;state.following=false;state.clock=0;state.eventLimit=300;arena.select(null);arena.setFrame(null);$('#arena-empty').hidden=false;$('#stop-match').hidden=true;renderFrame();renderTransport();
}
function adoptSession(session){
  state.session=session;state.following=session.state==='running'&&(session.metadata?.source==='training'||session.kind==='training'||session.metadata?.active===true);
  const metadata=session.metadata??{};
  $('#match-title').textContent=metadata.name??metadata.checkpoint?.split('/').pop()??state.selectedPath?.split('/').pop()??'Checkpoint exhibition';
  $('#match-source').textContent=sourceLabel(metadata.source??session.kind??'frontier').toUpperCase();
  renderSession();renderDetails();
}
function renderSession(){
  const session=state.session;if(!session)return;
  const status=$('#match-status');status.textContent=session.state==='running'?(session.metadata?.source==='training'?(session.metadata.writer_active===false?'WAITING FOR WRITER':'LIVE RECORDING'):'RUNNING'):String(session.state).toUpperCase();status.className=`status-pill ${session.state}`;
  const metadata=session.metadata??{},parts=[];
  if(metadata.seed!=null)parts.push(`Seed ${metadata.seed}`);
  if(metadata.archetype)parts.push(pretty(metadata.archetype));
  if(metadata.opponent)parts.push(`vs ${pretty(typeof metadata.opponent==='object'?metadata.opponent.archetype??metadata.opponent.strategy??'scripted':metadata.opponent)}`);
  parts.push(`${session.frame_count??state.frames.length} frames`);
  if(session.error)parts.push(session.error);
  $('#match-subtitle').textContent=parts.join(' · ');
  $('#stop-match').hidden=session.state!=='running';$('#stop-match').textContent=metadata.source==='training'?'Detach training view':session.kind==='exhibition'?'Stop exhibition':'Stop replay';renderDetails();
  $('#follow-live').disabled=!state.frames.length;
  if(session.state==='error'&&!state.frames.length){$('#arena-empty h3').textContent='Match could not be loaded';$('#arena-empty p').textContent=session.error??'Check the match format and checkpoint compatibility.';}
  else if(!state.frames.length){$('#arena-empty h3').textContent='Preparing the arena';$('#arena-empty p').textContent='The first simulator frame will appear here when it is ready.';}
}
async function pollFrames(){
  if(!state.session)return;
  const generation=state.generation,id=state.session.id;
  // A request from an old session must never hold up the newly selected one.
  if(state.polling===generation)return;
  state.polling=generation;
  try{
    let batches=0,more=true;
    while(more&&batches++<10){
      const after=state.frames.length?state.frames[state.frames.length-1].index:-1;
      const data=await api(`/api/matches/${encodeURIComponent(id)}/frames?after=${after}&limit=200`);
      if(generation!==state.generation)return;
      const frames=data.frames??[];if(data.status)state.session=data.status;
      for(const frame of frames)if(!state.frames.length||frame.index>state.frames[state.frames.length-1].index)state.frames.push(frame);
      more=frames.length===200;
      if(frames.length){
        if(state.following)state.position=state.frames.length-1;
        else if(state.frames.length===frames.length){state.position=0;state.clock=state.frames[0].time_s??0;}
        $('#arena-empty').hidden=true;renderFrame();
      }
      renderSession();renderTransport();
    }
  }catch(error){if(generation===state.generation)notify(`Match stream: ${error.message}`,true);}
  finally{if(state.polling===generation)state.polling=false;}
}

function cardToken(card,{next=false,selected=false}={}){
  const name=cardName(card);
  return `<div class="card-token${next?' next':''}${selected?' selected-card':''}" title="${esc(pretty(name))}${next?' · next card':''}"><span class="card-fallback">${esc(pretty(name))}</span><img src="/api/card-icon?name=${encodeURIComponent(name)}" alt="${esc(pretty(name))}" loading="lazy"></div>`;
}
function handleBrokenImages(container){container.querySelectorAll('img').forEach(img=>{img.addEventListener('error',()=>img.remove(),{once:true});img.addEventListener('load',()=>{const fallback=img.previousElementSibling;if(fallback)fallback.hidden=true;},{once:true});if(img.complete&&img.naturalWidth&&img.previousElementSibling)img.previousElementSibling.hidden=true;});}
function renderPlayer(id,container){
  const frame=currentFrame(),player=playerFor(frame,id);
  if(!player){container.innerHTML=`<div class="player-label"><strong>${id===0?'LEARNER':'OPPONENT'}</strong><span>WAITING</span></div><div class="hand"></div><div class="elixir-box"><div class="elixir-value">— <span>/ 10</span></div><div class="elixir-track"></div></div>`;return;}
  const elixir=Number(player.elixir),selected=id===0?frame?.policy?.selected_slot:null;
  container.innerHTML=`<div class="player-label"><strong>${id===0?'LEARNER':'OPPONENT'}</strong><span>P${id} · ${player.deck?.length??8} CARDS</span></div><div class="hand">${(player.hand??[]).map((card,index)=>cardToken(card,{selected:index===selected})).join('')}${player.next_card?cardToken(player.next_card,{next:true}):''}</div><div class="elixir-box"><div class="elixir-value">${Number.isFinite(elixir)?elixir.toFixed(1):'—'} <span>/ 10</span></div><div class="elixir-track"><span style="width:${Number.isFinite(elixir)?Math.max(0,Math.min(100,elixir*10)):0}%"></span></div></div>`;
  handleBrokenImages(container);
}
function renderFrame(){
  const frame=currentFrame();arena.setFrame(frame);renderPlayer(1,$('#opponent-state'));renderPlayer(0,$('#learner-state'));
  $('#match-clock').textContent=clock(frame?.time_s??0);$('#score-red').textContent=playerFor(frame,1)?.crowns??0;$('#score-blue').textContent=playerFor(frame,0)?.crowns??0;
  const phase=$('#arena-phase');phase.hidden=!frame?.terminal&&!frame?.phase;phase.textContent=frame?.terminal?(frame.winner==null||Number(frame.winner)<0?'Match drawn':`${Number(frame.winner)===0?'Learner':'Opponent'} wins`):pretty(frame?.phase);
  $('#snapshot').innerHTML=[['Tick',frame?.tick],['Entities',frame?.entities?.length],['Reward',frame?.reward],['Phase',frame?.phase]].map(([label,value])=>`<div><span>${label}</span><strong>${esc(typeof value==='number'?formatMetric(value):pretty(value))}</strong></div>`).join('');
  renderEntity();renderActions();if(state.inspector==='policy')renderPolicy();if(state.inspector==='events')renderEvents();renderTransport();
}
function renderEntity(){
  const frame=currentFrame(),entity=frame?.entities?.find(item=>String(item.uid)===String(arena.selected));
  if(!entity){$('#entity-detail').className='empty-detail';$('#entity-detail').textContent='Click a unit or tower on the arena to inspect it.';return;}
  $('#entity-detail').className='';
  const rows=[['ID',entity.uid],['Owner',Number(entity.owner)===0?'Learner · P0':'Opponent · P1'],['Kind',entity.kind],['Role',entity.role],['Position',`${Number(entity.x).toFixed(2)}, ${Number(entity.y).toFixed(2)}`],['Health',`${formatMetric(entity.hp)} / ${formatMetric(entity.max_hp)}`],['Target',entity.target_uid??'None'],['Range',finite(entity.range)?`${entity.range.toFixed(2)} tiles`:'—'],['Radius',finite(entity.radius)?`${entity.radius.toFixed(2)} tiles`:'—'],['Status',[...(entity.statuses??[]),...(entity.deploying?['Deploying']:[])].join(', ')||'Active']];
  $('#entity-detail').innerHTML=`<div class="entity-title">${cardToken(entity.card_id)}<span>${esc(pretty(entity.card_id))}</span></div><div class="health-track"><span style="width:${entity.max_hp>0?Math.max(0,Math.min(100,entity.hp/entity.max_hp*100)):0}%;background:${Number(entity.owner)===0?'var(--blue)':'var(--red)'}"></span></div><dl class="detail-list">${detailRows(rows)}</dl>`;handleBrokenImages($('#entity-detail'));
}
function detailRows(rows){return rows.filter(([,value])=>value!==undefined).map(([key,value])=>`<div><dt>${esc(key)}</dt><dd>${esc(typeof value==='object'?JSON.stringify(value):pretty(value))}</dd></div>`).join('');}
function actionText(action){const kind=String(action.kind??'action');if(/wait|noop/i.test(kind))return 'Wait';const card=action.card_id??(action.slot!=null?`Slot ${Number(action.slot)+1}`:'Card');return `${pretty(card)}${action.col!=null&&action.row!=null?` → (${action.col}, ${action.row})`:''}`;}
function renderActions(){const actions=currentFrame()?.actions??[];$('#frame-actions').className=actions.length?'':'empty-detail';$('#frame-actions').innerHTML=actions.length?actions.map(action=>`<div class="action-row"><span class="dot ${Number(action.player)===0?'blue':'red'}"></span><div>${esc(actionText(action))}<small>${Number(action.player)===0?'LEARNER':'OPPONENT'} · ${esc(pretty(action.kind))}</small></div></div>`).join(''):(currentFrame()?'No decision at this frame.':'No match selected.');}
function renderDetails(){
  const metadata=state.session?.metadata??{},container=$('#match-details'),wasOpen=container.querySelector('details')?.open??false;
  const basename=value=>typeof value==='string'?value.split('/').pop():undefined;
  const opponent=typeof metadata.opponent==='object'?metadata.opponent?.archetype??metadata.opponent?.strategy:metadata.opponent;
  const winner=metadata.result?.winner;
  const result=winner==null?undefined:Number(winner)<0?'Draw':Number(winner)===0?'Learner wins':'Opponent wins';
  const rows=[['Checkpoint',basename(metadata.checkpoint??metadata.learner_checkpoint)],['Opponent',opponent],['Opponent model',basename(metadata.opponent_checkpoint)],['Archetype',metadata.archetype??metadata.opponent?.archetype],['Seed',metadata.seed??metadata.match_seed],['Decision cap',metadata.max_decisions],['Result',result],['End reason',metadata.result?.reason]];
  container.innerHTML=`<dl class="detail-list">${detailRows(rows)}</dl><details class="match-raw-metadata"${wasOpen?' open':''}><summary>Technical metadata</summary><pre>${esc(JSON.stringify(metadata,null,2))}</pre></details>`;
}
function probability(value){return finite(value)?`${(value*100).toFixed(1)}%`:'—';}
function probabilityBar(label,value){return `<div class="policy-bar-row"><div class="policy-bar-label"><span>${esc(label)}</span><span>${probability(value)}</span></div><div class="policy-track"><span style="width:${finite(value)?Math.max(0,Math.min(100,value*100)):0}%"></span></div></div>`;}
function renderPolicy(){
  const frame=currentFrame(),policy=frame?.policy,container=$('#policy-detail');
  if(!policy){container.className='empty-detail';container.textContent='This frame has no policy trace. Historical action-only replays do not contain probabilities; launch a checkpoint exhibition or enable training capture.';return;}
  container.className='';const hand=playerFor(frame,0)?.hand??[];
  const unmasked = policy.distribution === 'unmasked' || policy.heatmap_available === false;
  const description = unmasked
    ? 'This historical trace stores raw model probabilities before illegal actions were removed. These values differ from the distribution used to choose legal actions. Only top placement cells were saved; a dense heatmap is unavailable.'
    : 'The heatmap shows where the learner would place a card if it chooses Play. It combines all playable hand cards, weighted by their card probabilities. The highlighted card is the sampled action.';
  container.innerHTML=`${probabilityBar(unmasked?'Raw Wait':'Wait',policy.wait_probability)}${probabilityBar(unmasked?'Raw Play':'Play',policy.play_probability)}<div class="section-label" style="margin-top:24px">${unmasked?'RAW HAND SLOT PROBABILITIES':'HAND SLOTS · GIVEN PLAY'}</div>${(policy.slot_probabilities??[]).map((value,index)=>`<div class="policy-slot">${cardToken(hand[index]??`Slot ${index+1}`,{selected:policy.selected_slot===index})}<div><div class="policy-bar-label"><span>${esc(pretty(cardName(hand[index]??`Slot ${index+1}`)))}</span><span>${probability(value)}</span></div><div class="policy-track"><span style="width:${finite(value)?Math.max(0,Math.min(100,value*100)):0}%"></span></div></div></div>`).join('')}<dl class="detail-list">${detailRows([['Selected slot',policy.selected_slot==null?'—':Number(policy.selected_slot)+1],['Action probability',probability(policy.action_probability)],['Entropy',formatMetric(policy.entropy)],['Placement cells',policy.heatmap?.length||'Unavailable']])}</dl>${policy.top_cells?.length?`<div class="section-label" style="margin-top:24px">RECORDED TOP PLACEMENTS · RAW</div><dl class="detail-list">${policy.top_cells.map((cells,index)=>`<div><dt>Slot ${index+1}</dt><dd>${esc((cells??[]).map(cell=>`(${cell.col}, ${cell.row}) · ${probability(cell.p)}`).join(' / '))}</dd></div>`).join('')}</dl>`:''}<p class="small muted" style="margin-top:16px">${description}</p>${unmasked?'':'<p class="small muted">Brighter cells have higher placement probability within this frame. Brightness is rescaled for each frame; the Wait/Play bars show how likely a play is. The map shows placement preference rather than win probability or an evaluation of the board.</p>'}`;handleBrokenImages(container);
}
function renderEvents(){
  const query=$('#event-search').value.toLowerCase(),items=[],seen=new Set();
  for(let position=0;position<=state.position;position++){
    const frame=state.frames[position];if(!frame)continue;
    for(const event of frame.events??[]){const key=`${event.tick}:${event.sequence}:${event.kind}:${JSON.stringify(event.data)}`;if(seen.has(key))continue;seen.add(key);items.push({position,time:frame.time_s,kind:pretty(event.kind),data:event.data});}
    for(const action of frame.actions??[])items.push({position,time:frame.time_s,kind:`${Number(action.player)===0?'Learner':'Opponent'} · ${actionText(action)}`,data:null});
  }
  const filtered=items.filter(item=>!query||`${item.kind} ${JSON.stringify(item.data)}`.toLowerCase().includes(query)),matches=filtered.slice(-state.eventLimit).reverse();
  const container=$('#event-list');container.innerHTML=matches.length?matches.map(item=>`<button class="event-item" data-frame-position="${item.position}"><time>${clock(item.time,true)}</time><span class="event-kind">${esc(item.kind)}</span>${item.data!=null?`<span class="event-data">${esc(typeof item.data==='string'?item.data:JSON.stringify(item.data))}</span>`:''}</button>`).join('')+(filtered.length>matches.length?`<button class="button secondary wide" id="older-events">Load ${Math.min(300,filtered.length-matches.length)} earlier events</button>`:''):'No events up to this frame.';
  container.querySelectorAll('[data-frame-position]').forEach(button=>button.addEventListener('click',()=>seekPosition(Number(button.dataset.framePosition))));
  $('#older-events')?.addEventListener('click',()=>{state.eventLimit+=300;renderEvents();});
}

function renderTransport(){
  const has=state.frames.length>0,first=state.frames[0]?.time_s??0,last=state.frames.at(-1)?.time_s??first,now=currentFrame()?.time_s??first;
  $('#scrubber').disabled=!has;$('#scrubber').min=String(first);$('#scrubber').max=String(Math.max(first+.001,last));$('#scrubber').value=String(now);
  const progress=last>first?(now-first)/(last-first)*100:0;$('#scrubber').style.background=`linear-gradient(to right,var(--accent) ${progress}%,#354356 ${progress}%)`;
  renderTimeline();
  $('#elapsed-label').textContent=clock(now,true);$('#duration-label').textContent=clock(last,true);
  $('#play-pause').disabled=!has;$('#play-pause').textContent=state.playing?'Ⅱ':'▶';$('#play-pause').setAttribute('aria-label',state.playing?'Pause':'Play');
  $('#step-back').disabled=!has||state.position===0;$('#step-forward').disabled=!has||state.position===state.frames.length-1;
  $('#frame-counter').textContent=`FRAME ${has?`${state.position+1} / ${state.frames.length}`:'—'}`;
  $('#follow-live').disabled=!has;$('#follow-live').classList.toggle('following',state.following);$('#follow-live').setAttribute('aria-pressed',String(state.following));
}
function renderTimeline(){
  const stamp=`${state.generation}:${state.frames.length}`;if(stamp===timelineStamp)return;timelineStamp=stamp;
  const first=state.frames[0]?.time_s??0,last=state.frames.at(-1)?.time_s??first,span=last-first||1;
  const markers=state.frames.flatMap((frame,position)=>(frame.actions??[]).filter(action=>!/wait|noop/i.test(String(action.kind))).map(action=>({position,action,time:frame.time_s})));
  $('#timeline-events').innerHTML=markers.map(({position,action,time})=>`<button class="timeline-event ${Number(action.player)===1?'opponent':''}" style="left:${Math.max(0,Math.min(100,(time-first)/span*100))}%" data-timeline-position="${position}" title="${esc(`${clock(time,true)} · ${Number(action.player)===0?'Learner':'Opponent'} · ${actionText(action)}`)}" aria-label="${esc(`${clock(time,true)} · ${Number(action.player)===0?'Learner':'Opponent'} · ${actionText(action)}`)}"></button>`).join('');
  $$('#timeline-events [data-timeline-position]').forEach(button=>button.addEventListener('click',()=>seekPosition(Number(button.dataset.timelinePosition))));
}
function seekPosition(position){if(!state.frames.length)return;state.playing=false;state.following=false;state.position=Math.max(0,Math.min(state.frames.length-1,position));state.clock=currentFrame().time_s??0;renderFrame();}
function seekTime(time){
  let low=0,high=state.frames.length-1;while(low<high){const mid=Math.ceil((low+high)/2);if((state.frames[mid].time_s??0)<=time)low=mid;else high=mid-1;}return low;
}
function togglePlay(){
  if(!state.frames.length)return;state.following=false;
  if(!state.playing&&state.position===state.frames.length-1&&state.session?.state!=='running'){state.position=0;renderFrame();}
  state.clock=currentFrame()?.time_s??0;state.playing=!state.playing;renderTransport();
}
function followLive(){if(!state.frames.length)return;state.following=!state.following;state.playing=false;if(state.following){state.position=state.frames.length-1;state.clock=currentFrame()?.time_s??0;renderFrame();}renderTransport();}
function animation(now){
  const elapsed=lastAnimation?Math.min((now-lastAnimation)/1000,.2):0;lastAnimation=now;
  if(state.playing&&state.frames.length){state.clock=Math.min(state.clock+elapsed*state.speed,state.frames.at(-1).time_s??state.clock);const position=seekTime(state.clock);if(position!==state.position){state.position=position;renderFrame();}if(state.position===state.frames.length-1&&state.session?.state!=='running'){state.playing=false;renderTransport();}}
  requestAnimationFrame(animation);
}

function setPage(page){state.page=page;$$('[data-page]').forEach(button=>{const active=button.dataset.page===page;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));});$$('.page').forEach(panel=>panel.classList.toggle('active',panel.id===`${page}-page`));if(page==='training')loadTraining();if(page==='compare')loadComparison();}
async function fetchRun(id){const run=await api(`/api/runs/${encodeURIComponent(id)}`);state.runCache.set(id,run);return run;}
async function loadTraining(){
  const id=$('#training-run').value;if(!id)return;const generation=++state.runGeneration.training;const container=$('#training-body');state.charts.training?.();container.className='analytics-empty';container.innerHTML='<div class="loading-skeleton" style="width:100%;height:240px"></div>';
  try{const run=await fetchRun(id);if(generation!==state.runGeneration.training)return;state.charts.training=renderRunAnalytics(container,[run],openCheckpoint);}
  catch(error){if(generation!==state.runGeneration.training)return;container.innerHTML=`<h2>Could not load this run</h2><p>${esc(error.message)}</p>`;}
}
async function loadComparison(){
  const a=$('#compare-run-a').value,b=$('#compare-run-b').value;if(!a||!b)return;const generation=++state.runGeneration.compare,container=$('#compare-body');state.charts.compare?.();container.className='analytics-empty';container.innerHTML='<div class="loading-skeleton" style="width:100%;height:240px"></div>';
  try{const runs=await Promise.all([fetchRun(a),fetchRun(b)]);if(generation!==state.runGeneration.compare)return;state.charts.compare=renderRunAnalytics(container,runs,openCheckpoint);}
  catch(error){if(generation!==state.runGeneration.compare)return;container.innerHTML=`<h2>Could not compare these runs</h2><p>${esc(error.message)}</p>`;}
}
function showLaunch(checkpoint){
  $('#launch-error').hidden=true;
  if(!(state.catalog.checkpoints??[]).length){$('#launch-error').textContent='No local V4 checkpoints were found. Save a training checkpoint, then refresh.';$('#launch-error').hidden=false;}
  if(checkpoint){const select=$('#launch-checkpoint');if(![...select.options].some(opt=>opt.value===checkpoint))select.add(new Option(checkpoint.split('/').pop(),checkpoint));select.value=checkpoint;$('#launch-submit').disabled=false;}
  $('#launch-dialog').showModal();
}
function openCheckpoint(path){setPage('watch');showLaunch(path);}
function parseDeck(input,selectId){
  const custom=$(input).value.trim();
  if(custom){const cards=custom.split(',').map(name=>name.trim()).filter(Boolean);if(cards.length!==8)throw new Error('A custom deck must contain exactly eight comma-separated card names.');if(new Set(cards).size!==8)throw new Error('A deck must contain eight distinct cards.');return cards;}
  const id=$(selectId).value;return id?(state.catalog.decks??[]).find(deck=>String(deck.id)===id)?.cards:undefined;
}
async function launch(event){
  event.preventDefault();const form=new FormData(event.target),button=$('#launch-submit');button.disabled=true;button.textContent='Starting match…';$('#launch-error').hidden=true;
  try{
    const body={checkpoint:form.get('checkpoint'),opponent:form.get('opponent'),archetype:$('#launch-archetype').value||'deterministic-cycle',seed:Number(form.get('seed')),max_decisions:Number(form.get('max_decisions'))};
    if(body.opponent==='checkpoint'){body.opponent_checkpoint=form.get('opponent_checkpoint');if(!body.opponent_checkpoint)throw new Error('Select an opponent checkpoint.');}
    body.learner_deck=parseDeck('#learner-custom-deck','#launch-learner-deck');body.opponent_deck=parseDeck('#opponent-custom-deck','#launch-opponent-deck');
    const vocabulary=state.catalog.learner_deck?.map(cardName);
    for(const [label,deck] of [['Learner',body.learner_deck],...(body.opponent==='checkpoint'?[['Checkpoint opponent',body.opponent_deck]]:[])])if(deck&&vocabulary?.length&&(deck.length!==vocabulary.length||deck.some(card=>!vocabulary.includes(cardName(card)))))throw new Error(`${label} must use the checkpoint’s eight supported cards. See the card vocabulary below Custom decks.`);
    const session=await api('/api/exhibitions',{method:'POST',body:JSON.stringify(body)});
    state.generation++;state.selectedPath=null;resetMatch();adoptSession(session);$('#launch-dialog').close();setPage('watch');renderLibrary();await pollFrames();loadLibrary();
  }catch(error){$('#launch-error').textContent=error.message;$('#launch-error').hidden=false;}
  finally{button.disabled=false;button.textContent='Launch exhibition →';}
}
async function stopMatch(){const id=state.session?.id,generation=state.generation;if(!id)return;$('#stop-match').disabled=true;try{const status=await api(`/api/matches/${encodeURIComponent(id)}/stop`,{method:'POST'});if(generation===state.generation){state.session=status;renderSession();await pollFrames();}}catch(error){notify(error.message,true);}finally{$('#stop-match').disabled=false;}}

$$('[data-page]').forEach(button=>button.addEventListener('click',()=>setPage(button.dataset.page)));
$$('[data-source]').forEach(button=>button.addEventListener('click',()=>{state.source=button.dataset.source;$$('[data-source]').forEach(item=>item.classList.toggle('active',item===button));renderLibrary();}));
$$('[data-inspector]').forEach(button=>button.addEventListener('click',()=>{state.inspector=button.dataset.inspector;$$('[data-inspector]').forEach(item=>item.classList.toggle('active',item===button));$$('.inspector-content').forEach(panel=>panel.classList.toggle('active',panel.id===`${state.inspector}-inspector`));if(state.inspector==='events')renderEvents();if(state.inspector==='policy')renderPolicy();}));
$('#match-search').addEventListener('input',event=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.query=event.target.value.trim().toLowerCase();renderLibrary();},150);});
$('#event-search').addEventListener('input',renderEvents);
$('#refresh-library').addEventListener('click',()=>{loadLibrary();loadCatalog();if(state.page==='training')loadTraining();if(state.page==='compare')loadComparison();});
for(const id of ['#launch-button','#empty-launch'])$(id).addEventListener('click',()=>showLaunch());
for(const id of ['#help-button','#recording-help','#keyboard-help'])$(id).addEventListener('click',()=>$('#help-dialog').showModal());
$$('[data-close-dialog]').forEach(button=>button.addEventListener('click',()=>button.closest('dialog').close()));
$$('dialog').forEach(dialog=>dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}}));
$('#launch-form').addEventListener('submit',launch);
$('#launch-opponent').addEventListener('change',event=>{const checkpoint=event.target.value==='checkpoint';$('#opponent-checkpoint-field').hidden=!checkpoint;$('#launch-opponent-checkpoint').required=checkpoint;$('#launch-archetype').disabled=checkpoint;});
$('#fit-arena').addEventListener('click',()=>arena.select(null));
for(const name of ['grid','labels','targets','ranges','heatmap'])$(`#overlay-${name}`).addEventListener('change',event=>{arena.setOption(name,event.target.checked);if(name==='heatmap'&&event.target.checked&&!currentFrame()?.policy?.heatmap?.length)notify('This frame has no placement probability data. Policy heatmaps are available in policy-recorded matches.');});
$('#play-pause').addEventListener('click',togglePlay);$('#step-back').addEventListener('click',()=>seekPosition(state.position-1));$('#step-forward').addEventListener('click',()=>seekPosition(state.position+1));$('#scrubber').addEventListener('input',event=>seekPosition(seekTime(Number(event.target.value))));$('#playback-speed').addEventListener('change',event=>{state.speed=Number(event.target.value);});$('#follow-live').addEventListener('click',followLive);$('#stop-match').addEventListener('click',stopMatch);
$('#training-run').addEventListener('change',loadTraining);$('#compare-run-a').addEventListener('change',loadComparison);$('#compare-run-b').addEventListener('change',loadComparison);
document.addEventListener('keydown', event => {
  const command = playbackCommand(event, {page: state.page, dialogOpen: !!$('dialog[open]')});
  if (!command) return;
  event.preventDefault();
  if (command.action === 'step') seekPosition(state.position + command.delta);
  else if (command.action === 'first') seekPosition(0);
  else if (command.action === 'last') seekPosition(state.frames.length - 1);
  else if (command.action === 'play') togglePlay();
  else if (command.action === 'follow') followLive();
  else if (command.action === 'clear') arena.select(null);
});
document.addEventListener('visibilitychange',()=>{if(!document.hidden){loadLibrary();pollFrames();}});

renderFrame();loadLibrary(true);loadCatalog();api('/api/geometry').then(geometry=>arena.setGeometry(geometry)).catch(error=>notify(`Using the standard arena layout: ${error.message}`,true));
setInterval(()=>{if(!document.hidden)loadLibrary();},10000);
setInterval(()=>{if(!document.hidden&&state.session&&(state.session.state==='running'||state.frames.length<(state.session.frame_count??0)))pollFrames();},800);
setInterval(()=>{if(!document.hidden&&state.page==='training'&&$('#training-run').value)loadTraining();},30000);
requestAnimationFrame(animation);
