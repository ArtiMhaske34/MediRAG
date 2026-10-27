import { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import {
  Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, BookOpen, Check,
  ChevronDown, CircleHelp, FileText, FlaskConical, FolderOpen, GitCompareArrows, HeartPulse,
  History, LayoutDashboard, Menu, MessageSquareText, Microscope, Pill, Plus, RefreshCw,
  Search, Send, Settings2, ShieldAlert, Sparkles, Stethoscope, ThumbsDown, ThumbsUp, Trash2,
  Upload, X
} from 'lucide-react';
import './App.css';
/* eslint react/prop-types: off */
const EMPTY_MESSAGES = [];

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const MODES = [
  ['auto','Automatic'],['search','Search records'],['calculate','Calculate BMI'],
  ['both','Search + BMI'],['check_interaction','Medicine pair demo'],['validate_labs','Lab value demo']
];
const ROUTES = {search:'Record search',calculate:'BMI calculation',both:'Search + BMI',check_interaction:'Limited interaction demo',validate_labs:'Hard-coded lab demo'};
const readSessions = () => { try { return JSON.parse(sessionStorage.getItem('medirag-sessions') || '[]'); } catch { return []; } };
const uid = () => crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random();
const formatSize = n => n < 1048576 ? Math.max(1,Math.round(n/1024))+' KB' : (n/1048576).toFixed(1)+' MB';

function App() {
  const [view,setView] = useState('overview'), [sessions,setSessions] = useState(readSessions);
  const [activeId,setActiveId] = useState(() => readSessions()[0]?.id || null);
  const [docs,setDocs] = useState([]), [health,setHealth] = useState('checking');
  const [mobile,setMobile] = useState(false), [toast,setToast] = useState(''), [busy,setBusy] = useState('');
  const [query,setQuery] = useState(''), [mode,setMode] = useState('auto'), [filter,setFilter] = useState(''), [source,setSource] = useState('');
  const [zip,setZip] = useState(null), [limit,setLimit] = useState('5'), [files,setFiles] = useState([]);
  const [pair,setPair] = useState(['','']), [compareQuery,setCompareQuery] = useState('Summarize the key findings in each selected record.'), [comparison,setComparison] = useState(null);
  const [evidence,setEvidence] = useState(true);
  const inputRef = useRef(null), endRef = useRef(null), zipRef = useRef(null), fileRef = useRef(null);
  const session = sessions.find(x => x.id === activeId) || null;
  const messages = session ? session.messages : EMPTY_MESSAGES;
  const lastAnswer = [...messages].reverse().find(x => x.role === 'assistant');
  const visibleDocs = useMemo(() => docs.filter(d => d.filename.toLowerCase().includes(filter.toLowerCase())),[docs,filter]);
  const demoCount = docs.filter(d => d.synthetic).length;

  useEffect(() => { sessionStorage.setItem('medirag-sessions',JSON.stringify(sessions)); },[sessions]);
  useEffect(() => { healthCheck(); refreshDocs(); const timer=setInterval(()=>{healthCheck();refreshDocs();},5000); return ()=>clearInterval(timer); },[]);
  useEffect(() => { endRef.current?.scrollIntoView({behavior:'smooth',block:'end'}); },[messages,busy]);
  useEffect(() => { if (!toast) return; const t=setTimeout(()=>setToast(''),3500); return ()=>clearTimeout(t); },[toast]);

  async function healthCheck() { try { const r=await axios.get(API+'/health',{timeout:3000}); const data=r.data||{}; setHealth(!data.model_loaded?'starting':!data.ollama_available?'ollama-offline':!data.llm_model_available?'model-missing':!data.embedding_model_available?'embedding-missing':'ready'); } catch { setHealth('offline'); } }
  async function refreshDocs() { try { const r=await axios.get(API+'/documents'); setDocs(Array.isArray(r.data.documents)?r.data.documents:[]); } catch { setDocs([]); } }
  function navigate(v) { setView(v); setMobile(false); }
  function newSession() { const s={id:uid(),title:'New analysis',messages:[],updatedAt:new Date().toISOString()}; setSessions(xs=>[s,...xs]); setActiveId(s.id); navigate('analysis'); setTimeout(()=>inputRef.current?.focus(),0); }
  function updateSession(id,fn) { setSessions(xs=>xs.map(s=>s.id===id?{...fn(s),updatedAt:new Date().toISOString()}:s)); }
  function addMessage(id,m,first) { updateSession(id,s=>({...s,title:s.messages.length===0&&first?first.slice(0,54):s.title,messages:[...s.messages,m]})); }
  function prompt(text,route) { setQuery(text); setMode(route); if (!session) newSession(); navigate('analysis'); setTimeout(()=>inputRef.current?.focus(),0); }
  async function ask(e) {
    e.preventDefault(); const text=query.trim(); if (!text||busy) return;
    let current=session;
    if(!current){ current={id:uid(),title:text.slice(0,54),messages:[],updatedAt:new Date().toISOString()}; setSessions(xs=>[current,...xs]); setActiveId(current.id); }
    addMessage(current.id,{id:uid(),role:'user',content:text,mode},text); setQuery(''); setBusy('analysis'); navigate('analysis');
    try { const r=await axios.post(API+'/chat',{query:text,mode,source:source||null}); addMessage(current.id,{id:uid(),role:'assistant',content:r.data.answer,route:r.data.route,sources:r.data.sources||[]}); setHealth('ready'); }
    catch(err){ addMessage(current.id,{id:uid(),role:'error',content:err.response?.data?.detail||'Could not reach the local service. Check the backend terminal and retry.'}); healthCheck(); }
    finally { setBusy(''); inputRef.current?.focus(); }
  }
  async function feedback(message,rating) {
    updateSession(activeId,s=>({...s,messages:s.messages.map(m=>m.id===message.id?{...m,feedback:rating}:m)}));
    try { await axios.post(API+'/feedback',{rating,session_id:activeId,response_id:message.id}); setToast('Feedback saved without question or answer text.'); }
    catch { setToast('Feedback marked in this tab; the API could not save it.'); }
  }
  async function importZip(e) {
    e.preventDefault(); if(!zip||busy)return; const data=new FormData(); data.append('file',zip); data.append('patient_limit',limit); setBusy('import');
    try { const r=await axios.post(API+'/documents/import-synthea',data); setToast('Imported '+(r.data.added?.length||0)+' new synthetic records; skipped '+(r.data.skipped_duplicates?.length||0)+' duplicates.'); setZip(null); if(zipRef.current)zipRef.current.value=''; await refreshDocs(); }
    catch(err){setToast(err.response?.data?.detail||'Import failed. Check the API terminal.');} finally{setBusy('');}
  }
  async function uploadDocs(e) {
    e.preventDefault(); if(!files.length||busy)return; const data=new FormData(); files.forEach(f=>data.append('files',f)); setBusy('upload');
    try { const r=await axios.post(API+'/documents/upload',data); setToast('Added '+(r.data.added?.length||0)+' new file(s); skipped '+(r.data.skipped_duplicates?.length||0)+' duplicates.'); setFiles([]); if(fileRef.current)fileRef.current.value=''; await refreshDocs(); }
    catch(err){setToast(err.response?.data?.detail||'Upload failed. Check the API terminal.');} finally{setBusy('');}
  }
  async function removeDoc(name) {
    if(!confirm('Remove '+name+' from the active library and rebuild the index?'))return; setBusy('delete');
    try{await axios.delete(API+'/documents/'+encodeURIComponent(name));await refreshDocs();setToast('Removed from the active index; a recoverable copy was kept.');}
    catch(err){setToast(err.response?.data?.detail||'Could not remove that file.');}finally{setBusy('');}
  }
  async function rebuild() {
    if(!confirm('Rebuild the index from every supported file? This may take a while.'))return;setBusy('reindex');
    try{const r=await axios.post(API+'/documents/reindex');await refreshDocs();setToast('Rebuilt from '+r.data.indexed_documents+' documents.');}
    catch(err){setToast(err.response?.data?.detail||'Index rebuild failed.');}finally{setBusy('');}
  }
  async function compare(e) {
    e.preventDefault();if(!pair[0]||!pair[1]||pair[0]===pair[1]){setToast('Choose two different records.');return;}setBusy('compare');
    try{const r=await axios.post(API+'/compare',{query:compareQuery,first_source:pair[0],second_source:pair[1]});setComparison(r.data);}
    catch(err){setToast(err.response?.data?.detail||'Comparison failed.');}finally{setBusy('');}
  }
  function clearHistory(){if(!confirm('Clear analysis history stored in this browser tab?'))return;setSessions([]);setActiveId(null);sessionStorage.removeItem('medirag-sessions');setToast('Tab history cleared.');}
  const names={overview:'Overview',analysis:'Analysis workspace',documents:'Document library',compare:'Compare records',methods:'Methods & limitations'};


  return <div className="app-shell">
    {mobile&&<button className="mobile-scrim" aria-label="Close navigation" onClick={()=>setMobile(false)}/>}
    <aside className={'sidebar '+(mobile?'sidebar-open':'')}>
      <div className="brand-lockup"><div className="brand-mark"><HeartPulse size={21}/></div><div><div className="brand-name">MediRAG</div><div className="brand-caption">Records intelligence workspace</div></div><button className="icon-button mobile-close" aria-label="Close navigation" onClick={()=>setMobile(false)}><X size={18}/></button></div>
      <button className="new-analysis" onClick={newSession}><Plus size={17}/><span>New analysis</span><span className="shortcut">⌘ K</span></button>
      <div className="nav-label">WORKSPACE</div><nav className="primary-nav" aria-label="Main navigation">
        <Nav icon={LayoutDashboard} label="Overview" active={view==='overview'} onClick={()=>navigate('overview')}/>
        <Nav icon={Activity} label="Analysis workspace" active={view==='analysis'} onClick={()=>navigate('analysis')}/>
        <Nav icon={FolderOpen} label="Document library" active={view==='documents'} count={docs.length} onClick={()=>navigate('documents')}/>
        <Nav icon={GitCompareArrows} label="Compare records" active={view==='compare'} onClick={()=>navigate('compare')}/>
      </nav>
      <div className="nav-label recent-label">RECENT SESSIONS <button className="tiny-action" onClick={clearHistory}>Clear</button></div>
      <div className="recent-list">{sessions.length?sessions.slice(0,7).map(s=><button key={s.id} className={'recent-item '+(s.id===activeId?'selected':'')} onClick={()=>{setActiveId(s.id);navigate('analysis')}}><History size={14}/><span>{s.title||'Untitled analysis'}</span></button>):<div className="recent-empty">Tab session history appears here.</div>}</div>
      <div className="sidebar-bottom"><div className="connection-card"><span className={'connection-dot '+health}/><div className="connection-copy"><strong>{health==='ready'?'Local model ready':health==='starting'?'Service is starting':health==='offline'?'Local API unavailable':health==='ollama-offline'?'Ollama is not running':health==='model-missing'?'phi3 model not found':health==='embedding-missing'?'Embedding model not found':'Checking local services'}</strong><span>API · localhost:8000</span></div><button className="plain-icon" aria-label="Refresh service status" onClick={healthCheck}><RefreshCw size={14}/></button></div>
        <button className={'nav-item methods-link '+(view==='methods'?'active':'')} onClick={()=>navigate('methods')}><Settings2 size={16}/><span>Methods & limitations</span></button>
        <div className="privacy-note"><ShieldAlert size={14}/><span>Educational prototype. No compliance certification.</span></div>
      </div>
    </aside>
    <main className="main-panel">
      <header className="topbar"><button className="icon-button mobile-menu" aria-label="Open navigation" onClick={()=>setMobile(true)}><Menu size={19}/></button><div className="breadcrumbs"><span>MediRAG</span><span className="crumb-divider">/</span><strong>{names[view]}</strong></div><div className="topbar-right"><div className="local-badge"><span className={'mini-dot '+health}/>{health==='ready'?'Local model ready':health==='offline'?'Local API offline':health==='ollama-offline'?'Start local Ollama':health==='model-missing'?'Local phi3 model missing':health==='embedding-missing'?'Local embedding model missing':'Checking local services'}</div>{demoCount>0&&<div className="demo-pill"><Sparkles size={13}/>Synthetic demo data · {demoCount}</div>}</div></header>
      <div className="view-content">
        {view==='overview'&&<Overview docs={docs} sessions={sessions} health={health} demoCount={demoCount} onStart={newSession} onNavigate={navigate} onPrompt={prompt}/>}
        {view==='analysis'&&<Analysis docs={docs} source={source} setSource={setSource} session={session} messages={messages} lastAnswer={lastAnswer} query={query} setQuery={setQuery} mode={mode} setMode={setMode} onSubmit={ask} busy={busy==='analysis'} inputRef={inputRef} endRef={endRef} onFeedback={feedback} onExport={()=>window.print()} evidence={evidence} setEvidence={setEvidence} onNew={newSession}/>}
        {view==='documents'&&<Documents docs={visibleDocs} filter={filter} setFilter={setFilter} zip={zip} setZip={setZip} limit={limit} setLimit={setLimit} zipRef={zipRef} onImport={importZip} files={files} setFiles={setFiles} fileRef={fileRef} onUpload={uploadDocs} onRemove={removeDoc} onRebuild={rebuild} busy={busy}/>}
        {view==='compare'&&<Compare docs={docs} pair={pair} setPair={setPair} query={compareQuery} setQuery={setCompareQuery} result={comparison} busy={busy==='compare'} onSubmit={compare}/>}
        {view==='methods'&&<Methods onNavigate={navigate}/>}
      </div>
      <footer className="app-footer"><span>For education and research. Review source material and use qualified clinical judgment.</span><span>History stays in this browser tab until it closes.</span></footer>
    </main>
    {toast&&<div className="toast" role="status"><Check size={16}/>{toast}<button aria-label="Dismiss" onClick={()=>setToast('')}><X size={14}/></button></div>}
  </div>;
}

function Nav({icon:Icon,label,active,onClick,count}){return <button className={'nav-item '+(active?'active':'')} onClick={onClick}><Icon size={17}/><span>{label}</span>{count>0&&<span className="nav-count">{count}</span>}</button>;}
function Overview({docs,sessions,health,demoCount,onStart,onNavigate,onPrompt}){
  const workflows=[
    [Search,'Search records','Find details in indexed records and review the returned passages.','RETRIEVAL','Which medications are listed in the selected record?','search'],
    [Activity,'Search + BMI','Retrieve a record and calculate BMI when weight and height are readable.','COMBINED ROUTE','Retrieve this record’s weight and height and calculate BMI.','both'],
    [GitCompareArrows,'Compare records','Ask the same question against two selected files.','SIDE BY SIDE','',''],
  ];
  return <section className="page overview-page">
    <div className="welcome-row"><div><div className="eyebrow"><span className="eyebrow-mark"><Activity size={13}/></span> MEDIRAG WORKSPACE</div><h1>Good work starts with<br/><span>clear evidence.</span></h1><p>Search local records, review returned evidence, and compare files.</p></div><div className="overview-art"><div className="art-ring ring-one"/><div className="art-ring ring-two"/><div className="art-center"><HeartPulse size={34}/></div><div className="art-label label-one"><FileText size={14}/>Source linked</div><div className="art-label label-two"><Activity size={14}/>Evidence reviewed</div></div></div>
    <div className="metric-grid"><Metric icon={FolderOpen} label="Indexed documents" value={docs.length} detail={demoCount?demoCount+' fictional demo records':'Manage the local source library'} tone="teal"/><Metric icon={MessageSquareText} label="Tab sessions" value={sessions.length} detail="Clears when this browser tab closes" tone="blue"/><Metric icon={Activity} label="API status" value={health==='ready'?'Ready':health==='offline'?'API offline':health==='ollama-offline'?'Ollama off':health==='model-missing'?'phi3 missing':'Starting'} detail="Local API and model status" tone={health==='ready'?'green':'amber'}/></div>
    <div className="section-title-row"><div><span className="eyebrow">WORKFLOWS</span><h2>Choose an analysis path</h2></div><button className="subtle-button" onClick={()=>onNavigate('methods')}>Methods & limits <ArrowUpRight size={15}/></button></div>
    <div className="workflow-grid">{workflows.map(([Icon,title,text,tag,sample,route])=><button className="workflow-card" key={title} onClick={()=>sample?onPrompt(sample,route):onNavigate('compare')}><div className="workflow-head"><span className="workflow-icon"><Icon size={19}/></span><ArrowUpRight size={16} className="workflow-open"/></div><span className="workflow-tag">{tag}</span><strong>{title}</strong><p>{text}</p></button>)}</div>
    <div className="overview-bottom"><div className="dataset-callout"><div className="callout-icon"><Sparkles size={19}/></div><div><strong>{demoCount?'Synthetic demo records are loaded':'Start with a reproducible demo'}</strong><p>{demoCount?'Fictional records are useful for pipeline demos, not clinical accuracy claims.':'Import a small Synthea FHIR R4 sample from the ZIP you downloaded.'}</p></div><button className="text-action" onClick={()=>onNavigate('documents')}>{demoCount?'View library':'Open library'} <ArrowRight size={15}/></button></div><div className="action-card"><div className="action-card-head"><span className="eyebrow">QUICK ACTION</span><button className="plain-icon" title="Start a fresh session" onClick={onStart}><Plus size={17}/></button></div><h3>Start a new analysis</h3><p>Select a request route and ask a question about your indexed records.</p><button className="primary-action" onClick={onStart}>Open workspace <ArrowRight size={15}/></button></div></div>
  </section>;
}
function Metric({icon:Icon,label,value,detail,tone}){return <div className="metric-card"><div className={'metric-icon '+tone}><Icon size={17}/></div><div className="metric-main"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div></div>;}


function Analysis({docs,source,setSource,session,messages,lastAnswer,query,setQuery,mode,setMode,onSubmit,busy,inputRef,endRef,onFeedback,onExport,evidence,setEvidence,onNew}){
  const setPrompt=(text,route)=>{setQuery(text);setMode(route);};
  const sources=lastAnswer?.sources||[];
  return <section className="page analysis-page"><div className="analysis-main">
    <div className="page-heading"><div><div className="eyebrow">ANALYSIS WORKSPACE</div><h1>{session?.title||'Start an analysis'}</h1><p>Review the route and evidence returned with each response.</p></div><div className="heading-actions"><button className="outline-button" onClick={onExport} disabled={!messages.length}><ArrowDownToLine size={15}/> Export / print</button><button className="icon-square" title="New analysis" onClick={onNew}><Plus size={17}/></button></div></div>
    {messages.length===0?<div className="analysis-empty"><div className="empty-symbol"><Stethoscope size={24}/></div><h2>What would you like to examine?</h2><p>Choose a workflow, then ask about indexed records.</p><div className="starter-grid">
      <Starter icon={Search} title="Look up a record" sample="Which medications are listed in the selected record?" route="search" onClick={setPrompt}/>
      <Starter icon={Activity} title="Search and calculate" sample="Retrieve this record’s weight and height and calculate BMI." route="both" onClick={setPrompt}/>
      <Starter icon={Pill} title="Medicine pair demo" sample="Check warfarin and aspirin in the demo table." route="check_interaction" onClick={setPrompt}/>
      <Starter icon={FlaskConical} title="Lab value demo" sample="Validate glucose 102 mg/dL using the demo interval." route="validate_labs" onClick={setPrompt}/>
    </div></div>:<div className="conversation-scroll">{messages.map(m=><Message key={m.id} message={m} onFeedback={onFeedback}/>)}{busy&&<div className="message-row assistant"><div className="assistant-avatar"><Stethoscope size={15}/></div><div className="message-body"><div className="message-meta">MediRAG · Processing</div><div className="loading-response"><i/><i/><i/>Searching evidence and preparing a response</div></div></div>}<div ref={endRef}/></div>}
    <div className="composer-area"><form className="composer" onSubmit={onSubmit}><div className="composer-toolbar"><label htmlFor="route-mode">Request route</label><span className="route-select-wrap"><select id="route-mode" value={mode} onChange={e=>setMode(e.target.value)}>{MODES.map(([id,label])=><option key={id} value={id}>{label}</option>)}</select><ChevronDown size={14}/></span><label className="source-filter" htmlFor="analysis-source">Source record</label><span className="source-select-wrap"><select id="analysis-source" value={source} onChange={e=>setSource(e.target.value)}><option value="">All records</option>{docs.map(d=><option key={d.filename} value={d.filename}>{d.filename}</option>)}</select><ChevronDown size={14}/></span><span className="route-hint">Choose one record for Search + BMI.</span></div><textarea ref={inputRef} value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();onSubmit(e);}}} placeholder="Ask about the indexed records…" rows={2} disabled={busy} aria-label="Ask MediRAG"/><div className="composer-bottom"><span className="composer-hint">Choose a source above to focus on one record.</span><button className="send-button" disabled={busy||!query.trim()}>{busy?<span className="button-spinner"/>:<Send size={15}/>}<span>{busy?'Working':'Analyze'}</span></button></div></form><div className="composer-disclaimer">Prototype output can be wrong. Review the source material; this is not medical advice.</div></div>
  </div><aside className={'evidence-panel '+(evidence?'':'evidence-hidden')}><div className="evidence-header"><div><span className="panel-kicker">RESPONSE SUPPORT</span><h2>Evidence & route</h2></div><button className="plain-icon" aria-label="Hide evidence" onClick={()=>setEvidence(false)}><X size={16}/></button></div>
    {lastAnswer?<><div className="route-result"><span className="route-result-icon"><Activity size={16}/></span><div><small>BACKEND ROUTE</small><strong>{ROUTES[lastAnswer.route]||'Unknown route'}</strong></div></div>{sources.length?<><div className="evidence-countline"><span>{sources.length} returned item{sources.length===1?'':'s'}</span><span>Latest response</span></div><div className="evidence-list">{sources.map((s,i)=><Evidence key={s.filename+i} source={s}/>)}</div></>:<EvidenceEmpty title="No source material returned" text="The API did not include evidence for this answer."/>}</>:<EvidenceEmpty title="Evidence appears here" text="After an answer arrives, review the text returned by the API."/>}
    <div className="evidence-caveat"><ShieldAlert size={15}/><span>Tool results are labeled separately from record passages. Demo ranges and interaction checks are limited.</span></div></aside>{!evidence&&<button className="evidence-reopen" onClick={()=>setEvidence(true)}><BookOpen size={15}/> Show evidence</button>}</section>;
}
function Starter({icon:Icon,title,sample,route,onClick}){return <button className="starter-card" onClick={()=>onClick(sample,route)}><span className="starter-icon"><Icon size={17}/></span><span><strong>{title}</strong><small>{sample}</small></span><ArrowUpRight size={15} className="starter-arrow"/></button>;}
function Message({message,onFeedback}){
 const user=message.role==='user';
 return <div className={'message-row '+message.role}>{message.role==='assistant'&&<div className="assistant-avatar"><Stethoscope size={15}/></div>}{message.role==='error'&&<div className="assistant-avatar error-avatar"><CircleHelp size={15}/></div>}<div className={'message-body '+(user?'user-body':'')}><div className="message-meta">{user?'You':message.role==='error'?'Service notice':'MediRAG'}{user&&<span> · {(MODES.find(x=>x[0]===message.mode)||[])[1]}</span>}{message.role==='assistant'&&<span> · {ROUTES[message.route]||'AI response'}</span>}</div><div className={'message-content '+message.role}>{message.content}</div>{message.role==='assistant'&&<div className="message-actions"><span><FileText size={13}/> {(message.sources||[]).length} returned source(s)</span><div className="feedback-actions"><span>Useful?</span><button className={message.feedback==='helpful'?'chosen':''} aria-label="Mark helpful" onClick={()=>onFeedback(message,'helpful')}><ThumbsUp size={14}/></button><button className={message.feedback==='unhelpful'?'chosen':''} aria-label="Mark needs review" onClick={()=>onFeedback(message,'unhelpful')}><ThumbsDown size={14}/></button></div></div>}</div></div>;
}
function Evidence({source}){return <article className="evidence-card"><div className="evidence-card-head"><span className={'file-icon '+(source.kind==='tool'?'tool-file-icon':'')}>{source.kind==='tool'?<Settings2 size={15}/>:<FileText size={15}/>}</span><span className="source-type">{source.kind==='tool'?'TOOL OUTPUT':'RECORD PASSAGE'}</span></div><h3>{source.filename}</h3><p>{source.content_snippet||'No preview returned.'}</p></article>;}
function EvidenceEmpty({title,text}){return <div className="evidence-empty"><div className="empty-icon"><FileText size={18}/></div><strong>{title}</strong><p>{text}</p></div>;}

function Documents({docs,filter,setFilter,zip,setZip,limit,setLimit,zipRef,onImport,files,setFiles,fileRef,onUpload,onRemove,onRebuild,busy}){
 return <section className="page documents-page"><div className="page-heading"><div><div className="eyebrow">DOCUMENT MANAGEMENT</div><h1>Document library</h1><p>Import a small Synthea sample or add local PDF, DOCX, and FHIR JSON files. Re-indexing rebuilds from the full source folder.</p></div><button className="outline-button" onClick={onRebuild} disabled={Boolean(busy)}><RefreshCw size={15}/> Rebuild index</button></div>
 <div className="import-grid"><div className="import-card synthea-card"><div className="import-card-top"><div className="import-icon demo"><Sparkles size={18}/></div><span className="feature-label">SYNTHETIC PATIENT DATA</span></div><h2>Import Synthea FHIR R4</h2><p>Choose the ZIP you downloaded. Import a small set first to keep indexing time manageable.</p><form onSubmit={onImport}><div className="form-row"><label className="select-file"><input ref={zipRef} type="file" accept=".zip,application/zip" onChange={e=>setZip(e.target.files?.[0]||null)}/><Upload size={15}/>{zip?zip.name:'Choose ZIP archive'}</label><label className="field-label">Records<select value={limit} onChange={e=>setLimit(e.target.value)}><option value="3">3 patients</option><option value="5">5 patients</option><option value="10">10 patients</option><option value="20">20 patients</option></select></label></div><button className="primary-action full" disabled={!zip||busy}>{busy==='import'?'Importing and indexing locally…':busy?'Wait for the current index task…':<>Import synthetic records <ArrowRight size={15}/></>}</button></form><div className="small-note"><ShieldAlert size={14}/> Fictional data supports demos, not clinical accuracy claims.</div></div>
 <div className="import-card"><div className="import-card-top"><div className="import-icon"><FolderOpen size={18}/></div><span className="feature-label">LOCAL DOCUMENTS</span></div><h2>Add files to the index</h2><p>Supported uploads are saved in the backend project folder and indexed locally.</p><form onSubmit={onUpload}><label className="drop-zone"><input ref={fileRef} type="file" multiple accept=".json,.pdf,.docx,application/pdf,application/json" onChange={e=>setFiles(Array.from(e.target.files||[]))}/><Upload size={18}/><strong>{files.length?files.length+' file(s) selected':'Choose supported files'}</strong><small>FHIR JSON, PDF, DOCX · 60 MB total</small></label><button className="outline-button full" disabled={!files.length||busy}>{busy==='upload'?'Adding and indexing locally…':busy?'Wait for the current index task…':<>Add documents and index <ArrowRight size={15}/></>}</button></form><div className="small-note"><CircleHelp size={14}/> Uploads remain in the local project; this screen does not send them to a cloud service.</div></div></div>
 {busy&&<div className="indexing-status" role="status"><RefreshCw size={16} className="spin"/><span><strong>Local indexing in progress</strong><small>The API stays available while records are being indexed. Larger libraries can take longer.</small></span></div>}<div className="library-table-section"><div className="library-table-head"><div><span className="eyebrow">INDEX CONTENTS</span><h2>Available source files <span className="count-badge">{docs.length}</span></h2></div><div className="table-search"><Search size={15}/><input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Filter by file name"/></div></div>
 {docs.length?<div className="document-table-wrap"><table className="document-table"><thead><tr><th>Document</th><th>Type</th><th>Size</th><th>Collection</th><th/></tr></thead><tbody>{docs.map(d=><tr key={d.filename}><td><span className="table-doc-icon"><FileText size={16}/></span><span className="table-doc-name">{d.filename}</span></td><td><span className="type-chip">{d.type.toUpperCase()}</span></td><td>{formatSize(d.size_bytes)}</td><td>{d.synthetic?<span className="synthetic-chip"><Sparkles size={12}/>Synthea · synthetic</span>:<span className="plain-chip">Local source</span>}</td><td><button className="plain-icon danger-hover" title="Remove document" aria-label={'Remove '+d.filename} onClick={()=>onRemove(d.filename)} disabled={Boolean(busy)}><Trash2 size={15}/></button></td></tr>)}</tbody></table></div>:<div className="empty-library"><div className="empty-icon"><FolderOpen size={19}/></div><strong>{filter?'No matching files':'No documents indexed yet'}</strong><p>{filter?'Try a shorter filename search.':'Import the Synthea FHIR ZIP or add a PDF, DOCX, or FHIR JSON above.'}</p></div>}</div></section>;
}


function Compare({docs,pair,setPair,query,setQuery,result,busy,onSubmit}){
 const change=(index,value)=>setPair(current=>index===0?[value,current[1]]:[current[0],value]);
 return <section className="page compare-page"><div className="page-heading"><div><div className="eyebrow">RECORD REVIEW</div><h1>Compare records</h1><p>Ask the same question against two selected files. Review each source separately before interpreting the summary.</p></div><span className="review-badge"><GitCompareArrows size={16}/> SOURCE-SCOPED</span></div>
 <form className="compare-form" onSubmit={onSubmit}><div className="compare-select-grid">{[0,1].map((i)=><label className="field-label" key={i}>Record {i===0?'A':'B'}<span className="select-wrap"><select value={pair[i]} onChange={e=>change(i,e.target.value)}><option value="">Select a document</option>{docs.map(d=><option key={d.filename} value={d.filename}>{d.filename}</option>)}</select><ChevronDown size={15}/></span></label>)}</div><label className="field-label compare-question">Question to compare<textarea value={query} onChange={e=>setQuery(e.target.value)} rows={2}/></label><div className="compare-actions"><span>{docs.length<2?'Add at least two indexed records to compare.':'Each answer retrieves from its selected file.'}</span><button className="primary-action" disabled={busy||docs.length<2}>{busy?'Comparing…':'Compare selected records'} <ArrowRight size={15}/></button></div></form>
 {result?<div className="compare-results"><div className="section-title-row"><div><span className="eyebrow">MODEL-GENERATED SUMMARY</span><h2>Comparison</h2></div></div><div className="comparison-summary">{result.comparison}</div><div className="compare-record-grid">{result.records?.map((r,i)=><article className="compare-record" key={r.filename}><div className="record-card-title"><span className="record-letter">{i?'B':'A'}</span><div><small>SELECTED RECORD</small><strong>{r.filename}</strong></div></div><p>{r.answer}</p><EvidenceStack sources={r.sources||[]}/></article>)}</div><div className="small-note"><ShieldAlert size={14}/> Model summary may omit or misstate details. Review both passages.</div></div>:<div className="compare-tip"><Microscope size={18}/><div><strong>Record-level comparison</strong><p>The endpoint scopes evidence to each selected file. It does not infer chronology or clinical change unless the text supports it.</p></div></div>}
 </section>;
}
function EvidenceStack({sources}){return sources.length?<div className="source-stack"><strong><FileText size={13}/> Evidence returned</strong>{sources.map((s,i)=><blockquote key={s.filename+i}>{s.content_snippet}</blockquote>)}</div>:<div className="no-evidence-inline">No source passage was returned for this file.</div>;}


function Methods({onNavigate}){
 const items=[
  [Search,'Record search','RETRIEVAL','FAISS retrieves text chunks from the local index. Returned passages are shown beside answers when available.'],
  [Activity,'BMI route','DETERMINISTIC CALCULATION','Uses explicit kg and m/cm values or labeled values in retrieved text. It reports a calculation only.'],
  [GitCompareArrows,'Search + BMI','COMBINED','Retrieves record text and computes BMI only when both values can be extracted.'],
  [Pill,'Medicine interaction','LIMITED DEMO TABLE','A small hard-coded pair list; not a comprehensive medicine interaction database.'],
  [FlaskConical,'Lab validation','HARDCODED DEMO INTERVALS','A short fixed list for demonstration. Lab ranges vary and are not clinically validated here.'],
  [Settings2,'Local embeddings','OLLAMA · OFFLINE','Document indexing and query retrieval use the locally installed nomic-embed-text model through Ollama. No differential privacy, HIPAA compliance, or security certification is claimed.']
 ];
 return <section className="page methods-page"><div className="page-heading"><div><div className="eyebrow">PROJECT NOTES</div><h1>Methods & limitations</h1><p>What the prototype does, and what its outputs do not establish.</p></div><div className="methods-icon"><BookOpen size={20}/></div></div><div className="methods-grid">{items.map(([Icon,title,label,text])=><article className="method-card" key={title}><span className="method-icon"><Icon size={18}/></span><span className="method-label">{label}</span><h2>{title}</h2><p>{text}</p></article>)}</div><div className="methods-callout"><ShieldAlert size={17}/><div><strong>Prototype scope</strong><p>For education and research. Review source material and use qualified clinical judgment.</p></div></div><div className="methods-actions"><button className="outline-button" onClick={()=>onNavigate('documents')}>Manage records <ArrowRight size={15}/></button></div></section>;
}
export default App;


