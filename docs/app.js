const resultRows = [["LSTM",1.41,6.48,76.36],["DCRNN",1.49,8.12,80.69],["GConvGRU",1.48,7.65,80.15],["GConvLSTM",1.36,5.61,73.65],["GCLSTM",1.38,5.95,74.73],["LRGCN",1.38,5.79,74.73],["TGCN",1.38,6.04,74.73],["A3T-GCN",1.46,6.61,79.06],["AGCRN",1.36,5.61,73.65],["STConv",1.5,6.95,81.23],["KNOWCAST",1.02,3.04,54.25]];
const figures = [["1",6,"Conceptual roadmap"],["2",7,"Knowledge integration methodology"],["3",8,"KNOWCAST Seq2Seq architecture"],["4",10,"KITP core architecture"],["5",17,"OD–path–section tripartite graph"],["6",21,"Dynamic metro headways"],["7",23,"Bayesian optimization progress"],["8",24,"Convergence curves"],["9",25,"Validation metrics"],["10",26,"Multi-horizon OD predictions"],["11",27,"Long-tail segmented errors"],["12",30,"Learned parameter trajectories"],["13",30,"Breakpoint detection"],["14",30,"Section flow density and Sankey"],["15",31,"Estimated versus observed train loads"]];
const chart=document.getElementById('baseline-chart');
function renderChart(metric){const sorted=[...resultRows].sort((a,b)=>a[metric]-b[metric]);const max=Math.max(...sorted.map(r=>r[metric]));chart.innerHTML=sorted.map(r=>'<div class="bar-row '+(r[0]==='KNOWCAST'?'highlight':'')+'"><span>'+r[0]+'</span><div class="bar-track"><div class="bar-fill" style="--value:'+r[metric]/max*100+'%"></div></div><b>'+r[metric].toFixed(2)+(metric===3?'%':'')+'</b></div>').join('');chart.setAttribute('aria-label',['','Mean absolute error comparison','Root mean squared error comparison','Weighted mean absolute percentage error comparison'][metric]);}
renderChart(1);
document.querySelectorAll('[data-metric]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-metric]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});renderChart(Number(button.dataset.metric));}));
const tabs=[...document.querySelectorAll('[role=tab]')];
function activateTab(tab,focus=false,syncHash=true){
  const index=tabs.indexOf(tab);
  if(index<0)return;
  tabs.forEach(t=>{const selected=t===tab;t.setAttribute('aria-selected',String(selected));t.tabIndex=selected?0:-1;document.getElementById(t.getAttribute('aria-controls')).hidden=!selected;});
  selectFramework('ki-'+(index+1));
  if(syncHash){
    const hash='#'+tab.getAttribute('aria-controls');
    if(location.hash!==hash)history.replaceState(null,'',hash);
  }
  if(focus)tab.focus();
}
tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>activateTab(tab));tab.addEventListener('keydown',event=>{let next;if(['ArrowRight','ArrowDown'].includes(event.key))next=(index+1)%tabs.length;if(['ArrowLeft','ArrowUp'].includes(event.key))next=(index-1+tabs.length)%tabs.length;if(event.key==='Home')next=0;if(event.key==='End')next=tabs.length-1;if(next!==undefined){event.preventDefault();activateTab(tabs[next],true);}});});
const frameworkLinks=[...document.querySelectorAll('[data-framework-target]')];
const frameworkStageCards=[...document.querySelectorAll('.framework-stage-grid article, .framework-service-details > div')];
let selectedFramework=null;
let hoveredFramework=null;
function frameworkFamily(target){if(!target)return[];return target==='service-passage'||target==='service-frequency'?[target,'ki-5']:[target];}
function paintFramework(){
  const selected=new Set(frameworkFamily(selectedFramework));
  const hovered=new Set(hoveredFramework?frameworkFamily(hoveredFramework):[]);
  frameworkLinks.forEach(link=>{
    const key=link.dataset.frameworkTarget;
    link.classList.toggle('is-active',selected.has(key));
    link.classList.toggle('is-hovered',hovered.has(key));
    if(key===selectedFramework)link.setAttribute('aria-current','location');
    else link.removeAttribute('aria-current');
  });
  frameworkStageCards.forEach(card=>card.classList.toggle('is-target',card.id==='framework-'+selectedFramework));
}
function selectFramework(target){selectedFramework=target;paintFramework();}
function navigateFramework(link,recordHistory=true,scroll=true,focusDestination=false){
  const target=link.dataset.frameworkTarget;
  const match=/^ki-([1-5])$/.exec(target);
  if(match)activateTab(tabs[Number(match[1])-1],false,false);
  else if(target==='service-passage'||target==='service-frequency')activateTab(tabs[4],false,false);
  selectFramework(target);
  const hash=link.getAttribute('href');
  const destination=document.getElementById(hash.slice(1));
  if(recordHistory&&location.hash!==hash)history.pushState(null,'',hash);
  if(focusDestination&&destination)destination.focus({preventScroll:true});
  if(scroll&&destination)destination.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});
}
frameworkLinks.forEach(link=>{
  link.addEventListener('click',event=>{event.preventDefault();navigateFramework(link,true,true,event.detail===0);});
  link.addEventListener('pointerenter',()=>{hoveredFramework=link.dataset.frameworkTarget;paintFramework();});
  link.addEventListener('pointerleave',()=>{hoveredFramework=null;paintFramework();});
  link.addEventListener('focus',()=>{hoveredFramework=link.dataset.frameworkTarget;paintFramework();});
  link.addEventListener('blur',()=>{hoveredFramework=null;paintFramework();});
});
function restoreFrameworkHash(){
  hoveredFramework=null;
  const link=frameworkLinks.find(item=>item.getAttribute('href')===location.hash);
  if(!link){
    activateTab(tabs[0],false,false);
    selectFramework(null);
    return;
  }
  navigateFramework(link,false,false);
  requestAnimationFrame(()=>document.getElementById(location.hash.slice(1))?.scrollIntoView({block:'start'}));
}
window.addEventListener('popstate',restoreFrameworkHash);
window.addEventListener('hashchange',restoreFrameworkHash);
paintFramework();
restoreFrameworkHash();
const dialog=document.getElementById('figure-dialog');
document.querySelectorAll('[data-figure]').forEach(button=>button.addEventListener('click',()=>{const item=figures.find(f=>f[0]===button.dataset.figure);document.getElementById('dialog-title').textContent='Figure '+item[0]+' · '+item[2];const img=document.getElementById('dialog-image');img.src='assets/figure-'+item[0].padStart(2,'0')+'.png';img.alt='Figure '+item[0]+'. '+item[2];document.getElementById('dialog-caption').textContent='Original figure from the paper · page '+item[1];document.getElementById('dialog-source').href='paper.html?page='+item[1];dialog.showModal();}));
document.getElementById('close-figure').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
document.getElementById('copy-citation').addEventListener('click',async()=>{const text=document.getElementById('bibtex').textContent;const status=document.getElementById('copy-status');try{await navigator.clipboard.writeText(text);status.textContent='BibTeX copied.';}catch{const range=document.createRange();range.selectNodeContents(document.getElementById('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent='Citation selected. Press Ctrl+C (or Command+C) to copy.';}});

const siteHeader = document.getElementById('site-header');
const menuToggle = document.getElementById('menu-toggle');
function closeMenu() {
  siteHeader.classList.remove('menu-open');
  menuToggle.setAttribute('aria-expanded', 'false');
  menuToggle.setAttribute('aria-label', 'Open navigation');
}
menuToggle.addEventListener('click', () => {
  const isOpen = siteHeader.classList.toggle('menu-open');
  menuToggle.setAttribute('aria-expanded', String(isOpen));
  menuToggle.setAttribute('aria-label', isOpen ? 'Close navigation' : 'Open navigation');
  if (isOpen) siteHeader.querySelector('nav a')?.focus();
});
siteHeader.querySelectorAll('nav a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && siteHeader.classList.contains('menu-open')) {
    closeMenu();
    menuToggle.focus();
  }
});
window.addEventListener('resize', () => { if (window.innerWidth > 1150) closeMenu(); });
function updateReadingProgress() {
  const available = document.documentElement.scrollHeight - window.innerHeight;
  const value = available > 0 ? Math.min(100, window.scrollY / available * 100) : 0;
  siteHeader.style.setProperty('--scroll-progress', value + '%');
}
window.addEventListener('scroll', updateReadingProgress, { passive: true });
updateReadingProgress();
