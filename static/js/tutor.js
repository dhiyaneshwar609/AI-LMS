const state = {
  chats: [], folders: [], activeChatId: null, attachments: [], polling: null,
  pollingJobId: null,
  backgroundJobs: {},
  backgroundNotifications: [],
  dark: localStorage.getItem('ai-lms-theme') === 'dark', draggedChatId: null, learningMode: localStorage.getItem('ai-lms-learning-mode') === '1',
  expandedFolders: JSON.parse(localStorage.getItem('ai-lms-expanded-folders') || '{}')
};
const $ = id => document.getElementById(id);

window.addEventListener('DOMContentLoaded', async () => { applyTheme(); ensureQuestionEditStyles(); wireEvents(); await loadEverything(); });

function wireEvents() {
  wireLearningMode();
  $('newChatBtn').addEventListener('click', createNewChat);
  $('createFolderBtn').addEventListener('click', openFolderModal);
  $('sidebarToggle').addEventListener('click', toggleSidebar);
  $('mobileSidebarClose').addEventListener('click', closeSidebar);
  document.addEventListener('click', e => {
    if (window.innerWidth <= 800 && document.body.classList.contains('sidebar-open') && !e.target.closest('.sidebar') && !e.target.closest('#sidebarToggle')) closeSidebar();
  });
  $('themeBtn').addEventListener('click', toggleTheme);
  $('chatSearch').addEventListener('input', renderSidebar);
  $('attachBtn').addEventListener('click', () => $('fileInput').click());
  $('fileInput').addEventListener('change', () => { state.attachments = Array.from($('fileInput').files); renderAttachments(); });
  setupComposerDropzone();
  setupComposerViewport();

  $('composer').addEventListener('submit', async e => {
    e.preventDefault();

    // Stop the active background generation when the Stop button is pressed.
    // The current architecture tracks jobs with pollingJobId rather than
    // state.polling, so checking only state.polling would incorrectly send
    // another question instead of stopping the running job.
    if (state.polling || state.pollingJobId) {
      await stopGeneration();
    } else {
      await sendQuestion();
    }
  });
  $('questionInput').addEventListener('input', autoResize);
  $('questionInput').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('composer').requestSubmit(); } });
  document.querySelectorAll('.suggestion').forEach(btn => btn.addEventListener('click', () => { $('questionInput').value = btn.textContent.trim(); autoResize(); $('questionInput').focus(); }));
  $('profileMenuBtn').addEventListener('click', e => { e.stopPropagation(); $('profileMenu').classList.toggle('open'); });
  $('logoutBtn').addEventListener('click', async () => { await fetch('/api/logout', {method:'POST'}); location.href='/login'; });
  $('chatActionCancel').addEventListener('click', closeDeleteModal);
  $('chatActionConfirm').addEventListener('click', confirmChatDelete);
  $('renameCancel').addEventListener('click', closeRenameModal);
  $('renameSave').addEventListener('click', saveChatRename);
  $('folderCancel').addEventListener('click', closeFolderModal);
  $('folderSave').addEventListener('click', saveFolder);
  $('moveCancel').addEventListener('click', closeMoveModal);
  $('jumpLatest').addEventListener('click', () => scrollBottom(true));
  $('chatScroll').addEventListener('scroll', updateJumpButton, {passive:true});
  $('chatMenuPin').addEventListener('click', menuPin);
  $('chatMenuRename').addEventListener('click', menuRename);
  $('chatMenuDelete').addEventListener('click', menuDelete);
  document.addEventListener('click', e => {
    if (!$('profileMenu').contains(e.target) && e.target !== $('profileMenuBtn')) $('profileMenu').classList.remove('open');
    if (!$('chatMenu').contains(e.target) && !e.target.closest('.more-action')) closeChatMenu();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeChatMenu(); closeDeleteModal(); closeRenameModal(); closeFolderModal(); closeMoveModal(); closeYoutubePopover(); } });
  $('renameInput').addEventListener('keydown', e => { if (e.key === 'Enter') saveChatRename(); });
  $('folderInput').addEventListener('keydown', e => { if (e.key === 'Enter') saveFolder(); });
}

async function loadEverything() {
  const [cr, fr] = await Promise.all([
    fetch('/api/chats'),
    fetch('/api/folders')
  ]);

  if (!cr.ok) {
    location.href = '/login';
    return;
  }

  const cd = await cr.json();
  const fd = await fr.json();

  state.chats = cd.chats || [];
  state.folders = fd.folders || [];

  // Load the sidebar but do NOT automatically open the previous chat.
  // The Tutor page always starts on a fresh New Chat screen.
  state.activeChatId = null;
  renderSidebar();
  renderMessages([]);
  showWelcome();

  requestAnimationFrame(() => {
    $('questionInput').focus();
  });
}

function renderSidebar() {
  const root = $('sidebarContent'); root.innerHTML = '';
  const q = $('chatSearch').value.trim().toLowerCase();
  const all = state.chats.filter(c => !q || String(c.title||'').toLowerCase().includes(q) || (c.messages||[]).some(m => String(m.content||'').toLowerCase().includes(q)));
  const pinned = all.filter(c => c.pinned);
  const folders = state.folders;
  const inFolder = id => all.filter(c => String(c.folder_id||'') === String(id));

  root.appendChild(sectionHeading('Pinned', pinned.length));
  const pinnedBox = document.createElement('div'); pinnedBox.className='section-list';
  pinned.forEach(c => pinnedBox.appendChild(chatRow(c))); if (!pinned.length) pinnedBox.appendChild(emptyText(q ? 'No matching pinned chats.' : 'No pinned chats yet.')); root.appendChild(pinnedBox);

  root.appendChild(sectionHeading('Folders', folders.length));
  const folderBox = document.createElement('div'); folderBox.className='folder-list';
  if (!folders.length) folderBox.appendChild(emptyText('No folders yet.'));
  folders.forEach(folder => folderBox.appendChild(folderBlock(folder, inFolder(folder.id))));
  root.appendChild(folderBox);

  root.appendChild(sectionHeading('Recent Chats', all.filter(c => !c.folder_id && !c.pinned).length));
  const recent = all.filter(c => !c.folder_id && !c.pinned);
  const recentBox = document.createElement('div'); recentBox.className='section-list';
  recent.forEach(c => recentBox.appendChild(chatRow(c))); if (!recent.length) recentBox.appendChild(emptyText(q ? 'No matching recent chats.' : 'No conversations yet.')); root.appendChild(recentBox);
}

function sectionHeading(name, count) {
  const el = document.createElement('div'); el.className='sidebar-section-title';
  const left=document.createElement('span'); left.textContent=name;
  const right=document.createElement('span'); right.className='section-count'; right.textContent=String(count);
  el.append(left,right); return el;
}
function emptyText(text) { const e=document.createElement('div'); e.className='history-empty'; e.textContent=text; return e; }

function folderBlock(folder, chats) {
  const wrap=document.createElement('div'); wrap.className='folder-block';
  const head=document.createElement('div'); head.className='folder-row'; head.dataset.folderId=folder.id;
  const toggle=document.createElement('button'); toggle.className='folder-toggle'; toggle.type='button';
  const expanded=state.expandedFolders[String(folder.id)] !== false;
  toggle.textContent=expanded?'▾':'▸'; toggle.setAttribute('aria-label',expanded?'Collapse folder':'Expand folder');
  const icon=document.createElement('span'); icon.className='folder-icon'; icon.textContent='📂';
  const name=document.createElement('span'); name.className='folder-name'; name.textContent=folder.name;
  const count=document.createElement('span'); count.className='folder-count'; count.textContent=String(chats.length);
  toggle.addEventListener('click',e=>{e.stopPropagation(); state.expandedFolders[String(folder.id)] = !expanded; localStorage.setItem('ai-lms-expanded-folders',JSON.stringify(state.expandedFolders)); renderSidebar();});
  head.append(toggle,icon,name,count); head.addEventListener('dragover',e=>{e.preventDefault(); head.classList.add('drag-over');}); head.addEventListener('dragleave',()=>head.classList.remove('drag-over')); head.addEventListener('drop',async e=>{e.preventDefault(); head.classList.remove('drag-over'); const id=e.dataTransfer.getData('text/chat-id')||state.draggedChatId; if(id) await moveChat(id,folder.id);});
  wrap.appendChild(head);
  if(expanded){ const list=document.createElement('div'); list.className='folder-chat-list'; if(chats.length) chats.forEach(c=>list.appendChild(chatRow(c))); else list.appendChild(emptyText('Drop chats here')); wrap.appendChild(list); }
  return wrap;
}

function chatRow(chat) {
  const row=document.createElement('div');
  row.className='chat-row'+(String(chat.id)===String(state.activeChatId)?' active':'');
  row.draggable=true;
  row.dataset.chatId=chat.id;
  row.addEventListener('dragstart',e=>{
    state.draggedChatId=chat.id;
    e.dataTransfer.effectAllowed='move';
    e.dataTransfer.setData('text/chat-id',String(chat.id));
    row.classList.add('dragging');
  });
  row.addEventListener('dragend',()=>{
    state.draggedChatId=null;
    row.classList.remove('dragging');
  });

  const title=document.createElement('button');
  title.type='button';
  title.className='chat-title';
  title.title=chat.title||'New Chat';
  const titleText=document.createElement('span');
  titleText.className='chat-title-text';
  titleText.textContent=chat.title||'New Chat';
  title.appendChild(titleText);
  title.addEventListener('click',()=>selectChat(chat.id));

  const actions=document.createElement('div');
  actions.className='chat-actions';

  // Dedicated Move-to-folder button remains separate from the three-dot menu.
  const folderBtn=document.createElement('button');
  folderBtn.type='button';
  folderBtn.className='chat-action folder-action';
  folderBtn.textContent='📂';
  folderBtn.title='Move to folder';
  folderBtn.setAttribute('aria-label','Move to folder');
  folderBtn.addEventListener('click',e=>{
    e.stopPropagation();
    openMoveModal(chat.id);
  });

  // No star/pin button in the chat row. Pin remains available inside ⋯.
  const more=document.createElement('button');
  more.type='button';
  more.className='chat-action more-action';
  more.textContent='⋯';
  more.title='Chat options';
  more.setAttribute('aria-label','Chat options');
  more.addEventListener('click',e=>{
    e.stopPropagation();
    openChatMenu(chat,more);
  });

  actions.append(folderBtn,more);
  row.append(title,actions);
  requestAnimationFrame(()=>{
    if(titleText.scrollWidth > title.clientWidth + 4) title.classList.add('overflow');
  });
  return row;
}
let menuChatId=null, pendingDeleteChatId=null, renameChatId=null, moveChatId=null;
function openChatMenu(chat,anchor){ closeChatMenu(); menuChatId=chat.id; const m=$('chatMenu'); $('chatMenuPin').textContent=chat.pinned?'☆ Unpin chat':'☆ Pin chat'; const r=anchor.getBoundingClientRect(); const w=190,h=115; let left=r.right-w,top=r.bottom+4; if(left<8)left=8;if(left+w>innerWidth-8)left=innerWidth-w-8;if(top+h>innerHeight-8)top=r.top-h-4;if(top<8)top=8;m.style.left=left+'px';m.style.top=top+'px';m.classList.add('open');}
function closeChatMenu(){ $('chatMenu').classList.remove('open'); menuChatId=null; }
async function menuPin(){const c=state.chats.find(x=>String(x.id)===String(menuChatId));closeChatMenu();if(c)await togglePinChat(c);}
function menuRename(){const c=state.chats.find(x=>String(x.id)===String(menuChatId));closeChatMenu();if(c)openRenameModal(c);}
function menuDelete(){pendingDeleteChatId=menuChatId;closeChatMenu();if(pendingDeleteChatId)openDeleteModal();}
async function togglePinChat(chat){const r=await fetch(`/api/chats/${chat.id}/pin`,{method:'POST'});const d=await r.json();if(d.ok){chat.pinned=!!d.pinned;renderSidebar();}}

function openDeleteModal(){ $('chatActionModal').classList.add('open'); $('chatDeleteError').textContent=''; }
function closeDeleteModal(){ $('chatActionModal').classList.remove('open'); pendingDeleteChatId=null; }
async function confirmChatDelete(){const id=pendingDeleteChatId;if(!id)return;const r=await fetch(`/api/chats/${id}`,{method:'DELETE'});const d=await r.json();if(!d.ok){$('chatDeleteError').textContent=d.error||'Could not delete chat.';return;}state.chats=state.chats.filter(c=>String(c.id)!==String(id));closeDeleteModal();if(String(state.activeChatId)===String(id)){state.activeChatId=state.chats[0]?.id||null;if(state.activeChatId)await selectChat(state.activeChatId);else{renderMessages([]);showWelcome();}}renderSidebar();}

function openRenameModal(chat){renameChatId=chat.id;$('renameInput').value=chat.title||'';$('renameError').textContent='';$('renameModal').classList.add('open');requestAnimationFrame(()=>{$('renameInput').focus();$('renameInput').select();});}
function closeRenameModal(){$('renameModal').classList.remove('open');renameChatId=null;}
async function saveChatRename(){const id=renameChatId,title=$('renameInput').value.trim();if(!id||!title)return;const r=await fetch(`/api/chats/${id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title})});const d=await r.json();if(!d.ok){$('renameError').textContent=d.error||'Could not rename chat.';return;}const c=state.chats.find(x=>String(x.id)===String(id));if(c)c.title=d.chat.title;closeRenameModal();renderSidebar();}

function openFolderModal(){$('folderInput').value='';$('folderError').textContent='';$('folderModal').classList.add('open');requestAnimationFrame(()=>$('folderInput').focus());}
function closeFolderModal(){$('folderModal').classList.remove('open');}
async function saveFolder(){const name=$('folderInput').value.trim();if(!name)return;const r=await fetch('/api/folders',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});const d=await r.json();if(!d.ok){$('folderError').textContent=d.error||'Could not create folder.';return;}state.folders.unshift(d.folder);state.expandedFolders[String(d.folder.id)]=true;closeFolderModal();renderSidebar();}

function openMoveModal(chatId){moveChatId=chatId;const box=$('folderChoices');box.innerHTML='';const none=document.createElement('button');none.className='folder-choice';none.innerHTML='<span>📂</span><span>Remove from folder</span>';none.onclick=()=>moveChat(chatId,null);box.appendChild(none);if(!state.folders.length){box.appendChild(emptyText('No folders created yet.'));}else state.folders.forEach(f=>{const b=document.createElement('button');b.className='folder-choice';b.innerHTML=`<span>📂</span><span>${escapeHtml(f.name)}</span><span class="choice-count">${state.chats.filter(c=>String(c.folder_id)===String(f.id)).length}</span>`;b.onclick=()=>moveChat(chatId,f.id);box.appendChild(b);});$('moveModal').classList.add('open');}
function closeMoveModal(){$('moveModal').classList.remove('open');moveChatId=null;}
async function moveChat(chatId,folderId){const r=await fetch(`/api/chats/${chatId}/move`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder_id:folderId})});const d=await r.json();if(!d.ok){showToast(d.error||'Could not move chat.');return;}const c=state.chats.find(x=>String(x.id)===String(chatId));if(c)c.folder_id=folderId;closeMoveModal();renderSidebar();showToast(folderId?'Chat moved to folder.':'Chat removed from folder.');}

async function createNewChat(){const r=await fetch('/api/chats',{method:'POST'});const d=await r.json();if(!d.ok){showToast(d.error||'Could not create chat.');return;}state.chats.unshift(d.chat);state.activeChatId=d.chat.id;renderSidebar();renderMessages([]);showWelcome();$('questionInput').focus();}
async function selectChat(id){
  state.activeChatId=id;
  const r=await fetch(`/api/chats/${id}`);
  if(!r.ok)return;
  const d=await r.json();
  renderMessages(d.chat.messages||[]);
  renderSidebar();
  syncActiveGenerationUI();
}
function showWelcome(){$('welcome').style.display='';updateJumpButton();}
function hideWelcome(){$('welcome').style.display='none';}

function getRunningJobsForChat(chatId){
  return Object.values(state.backgroundJobs).filter(j=>j && j.chatId!=null && String(j.chatId)===String(chatId) && j.status==='running');
}
function getActiveRunningJob(){
  return getRunningJobsForChat(state.activeChatId)[0] || null;
}
function syncActiveGenerationUI(){
  const job=getActiveRunningJob();
  if(job){
    state.pollingJobId=job.jobId;
    setGenerating(true);
    if(!$('aiThinking')) showThinking();
  }else{
    if(state.pollingJobId && !getRunningJobsForChat(state.activeChatId).length){
      state.pollingJobId=null;
    }
    setGenerating(false);
    if($('aiThinking')) removeThinking(true);
  }
}

function renderMessages(messages){
  const box=$('messages');
  box.innerHTML='';
  if(!messages.length){
    showWelcome();
    syncActiveGenerationUI();
    return;
  }
  hideWelcome();
  messages.forEach((m,i)=>appendMessage(m.role,m.content,m.sources||[],m.id||'',i,messages,m.youtube||null,m.learning_mode||false));
  const running=getActiveRunningJob();
  if(running) showThinking();
  scrollBottom(false);
}
function actionIcon(name){
  const common='width=18 height=18 viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
  const paths={
    edit:`<svg ${common}><path d="M4 20h4l10.5-10.5a2.1 2.1 0 0 0-3-3L5 17v3Z"></path><path d="M14.5 7.5l2 2"></path></svg>`,
    copy:`<svg ${common}><rect x=8 y=8 width=11 height=11 rx=2></rect><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"></path></svg>`,
    retry:`<svg ${common}><path d="M20 11a8 8 0 0 0-14.9-4L3 10"></path><path d="M3 5v5h5"></path><path d="M4 13a8 8 0 0 0 14.9 4L21 14"></path><path d="M21 19v-5h-5"></path></svg>`,
    folder:`<svg ${common}><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4l2 2h7A2.5 2.5 0 0 1 21 9.5v7A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5z"></path><path d="M12 11v5M9.5 13.5h5"></path></svg>`,
    pdf:`<svg ${common}><path d="M6 3h8l4 4v14H6z"></path><path d="M14 3v5h5"></path><path d="M8.5 16h7M8.5 12.5h3"></path></svg>`,
    youtube:`<svg ${common}><rect x="3" y="6" width="18" height="12" rx="3"></rect><path d="M10 9l5 3-5 3V9Z" fill="currentColor" stroke="none"></path></svg>`,
    sources:`<svg ${common}><path d="M5 4.5h9a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3V4.5Z"></path><path d="M8 4.5v11.8c0 2 1.2 3.7 3.2 3.7"></path><path d="M8.5 8h5.5M8.5 11.5h5.5"></path></svg>`
  };
  return paths[name]||'';
}

function ensureQuestionEditStyles(){
  if(document.getElementById('questionEditStyles')) return;
  const style=document.createElement('style');
  style.id='questionEditStyles';
  style.textContent=`
    .message.user{position:relative}
    /* Invisible hover bridge: keeps Edit/Copy visible while the cursor moves
       from the question bubble up to the action buttons. */
    .message.user::after{content:"";position:absolute;right:0;top:-44px;width:100%;height:44px;pointer-events:auto;background:transparent;z-index:4}
    .question-actions{display:flex;gap:6px;position:absolute;right:0;top:-38px;opacity:0;pointer-events:none;transform:translateY(4px);transition:opacity .16s ease,transform .16s ease;z-index:5}
    .message.user:hover .question-actions,.message.user.editing .question-actions,.message.user:focus-within .question-actions{opacity:1;pointer-events:auto;transform:translateY(0)}
    .question-actions button{width:34px;height:34px;border:1px solid var(--border,#e5e7eb);border-radius:10px;background:var(--surface,#fff);color:var(--muted,#64748b);display:grid;place-items:center;cursor:pointer;box-shadow:0 5px 18px rgba(0,0,0,.08)}
    .question-actions button:hover{color:var(--text,#0f172a);background:var(--surface-2,#f8fafc)}
    .question-actions svg{width:17px;height:17px}
    .question-edit-box{min-width:280px;max-width:min(680px,calc(100vw - 80px));background:var(--surface,#fff);border:1px solid var(--border,#dbe1ea);border-radius:16px;padding:12px;box-shadow:0 10px 30px rgba(0,0,0,.10)}
    .question-edit-box textarea{display:block;width:100%;min-height:76px;max-height:180px;resize:vertical;border:0;outline:0;background:transparent;color:var(--text,#0f172a);font:inherit;line-height:1.5}
    .question-edit-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:8px}
    .question-edit-actions button{border:0;border-radius:9px;padding:7px 13px;font:inherit;cursor:pointer}
    .question-edit-cancel{background:transparent;color:var(--muted,#64748b)}
    .question-edit-send{background:#6d45e8;color:#fff}
    body.dark .question-actions button{background:#252733;border-color:#3a3d4b;color:#b8bdcc}
    body.dark .question-edit-box{background:#20222d;border-color:#3a3d4b}
    body.dark .question-edit-box textarea{color:#f4f6fb}
    @media(max-width:800px){
      .message.user::after{top:-48px;height:48px}
      .question-actions{top:-42px;right:0}
      .question-edit-box{max-width:calc(100vw - 42px);min-width:240px}
    }
  `;
  document.head.appendChild(style);
}

function appendMessage(role,content,sources=[],messageId='',index=0,messages=[],youtube=null,learningMode=false){
  hideWelcome();
  const wrap=document.createElement('article');
  wrap.className=`message ${role}`;
  if(messageId)wrap.dataset.messageId=messageId;
  wrap.dataset.messageIndex=String(index);

  const body=document.createElement('div'); body.className='message-body';
  const bubble=document.createElement('div'); bubble.className='message-bubble';

  if(role==='assistant'){
    bubble.innerHTML=formatAnswer(content||'');
    body.appendChild(bubble);
  }else{
    bubble.textContent=content||'';
    body.appendChild(bubble);

    const actions=document.createElement('div');
    actions.className='question-actions';
    actions.innerHTML=`<button type="button" data-qaction="copy" title="Copy" aria-label="Copy">${actionIcon('copy')}</button><button type="button" data-qaction="edit" title="Edit message" aria-label="Edit message">${actionIcon('edit')}</button>`;
    actions.querySelector('[data-qaction=copy]').onclick=()=>copyQuestion(content||'');
    actions.querySelector('[data-qaction=edit]').onclick=()=>startInlineQuestionEdit(wrap,content||'',Number(index));
    body.appendChild(actions);
  }

  if(role==='assistant'){
    if(sources.length){
      const src=document.createElement('div'); src.className='sources';
      src.textContent='Sources: '+uniqueSourceLabels(sources).join(' • ');
      body.appendChild(src);
    }
    const actions=document.createElement('div'); actions.className='answer-actions';
    actions.innerHTML=`<button data-action="copy" title="Copy" aria-label="Copy">${actionIcon('copy')}</button><button data-action="retry" title="Try again" aria-label="Try again">${actionIcon('retry')}</button><button data-action="folder" title="Add to folder" aria-label="Add to folder">${actionIcon('folder')}</button><button data-action="pdf" title="Generate as PDF" aria-label="Generate as PDF">${actionIcon('pdf')}</button><button data-action="sources" title="View sources" aria-label="View sources">${actionIcon('sources')}</button><button data-action="youtube" title="Recommended YouTube video" aria-label="Recommended YouTube video">${actionIcon('youtube')}</button>`;
    actions.querySelector('[data-action=copy]').onclick=()=>copyAnswer(content||'');
    actions.querySelector('[data-action=retry]').onclick=()=>retryAnswer(messageId);
    actions.querySelector('[data-action=folder]').onclick=()=>openMoveModal(state.activeChatId);
    actions.querySelector('[data-action=pdf]').onclick=()=>downloadAnswerPdf(state.activeChatId,messageId);
    actions.querySelector('[data-action=sources]').onclick=(e)=>showSourcesPopover(e.currentTarget,sources);
    actions.querySelector('[data-action=youtube]').onclick=(e)=>toggleYoutubeForMessage(e.currentTarget,messageId,content||'',youtube,learningMode);
    body.appendChild(actions);
  }

  wrap.appendChild(body);
  $('messages').appendChild(wrap);
}

async function copyQuestion(text){
  const value=String(text||'');
  try{
    if(navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(value);
    else{
      const ta=document.createElement('textarea');
      ta.value=value; ta.style.position='fixed'; ta.style.opacity='0';
      document.body.appendChild(ta); ta.focus(); ta.select();
      document.execCommand('copy'); ta.remove();
    }
    showToast('Question copied.');
  }catch(_){showToast('Could not copy the question.');}
}

function startInlineQuestionEdit(wrap,original,index){
  if(!wrap || wrap.classList.contains('editing')) return;
  wrap.classList.add('editing');
  const body=wrap.querySelector('.message-body'), oldBubble=wrap.querySelector('.message-bubble'), oldActions=wrap.querySelector('.question-actions');
  if(!body||!oldBubble)return;

  const box=document.createElement('div');
  box.className='question-edit-box';
  box.innerHTML=`<textarea aria-label="Edit message"></textarea><div class="question-edit-actions"><button type="button" class="question-edit-cancel">Cancel</button><button type="button" class="question-edit-send">Send</button></div>`;
  const textarea=box.querySelector('textarea'); textarea.value=original;
  oldBubble.style.display='none'; if(oldActions)oldActions.style.display='none'; body.appendChild(box);

  const cancel=()=>{
    box.remove(); oldBubble.style.display=''; if(oldActions)oldActions.style.display=''; wrap.classList.remove('editing');
  };

  const submit=async()=>{
    const edited=textarea.value.trim();
    if(!edited){showToast('Question cannot be empty.');textarea.focus();return;}
    if(edited===String(original).trim()){cancel();return;}

    // Editing is allowed while AI is thinking. Stop the current job first.
    if(state.polling || state.pollingJobId) await stopGeneration();
    clearInterval(state.polling); state.polling=null; state.pollingJobId=null;
    removeThinking(true); setGenerating(false);

    const chatId=state.activeChatId;
    if(!chatId){showToast('No active chat.');cancel();return;}

    const r=await fetch(`/api/chats/${chatId}/edit-message`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message_index:index,content:edited})
    });
    const d=await r.json().catch(()=>({}));
    if(!r.ok||!d.ok){showToast(d.error||'Could not edit the question.');return;}

    const local=state.chats.find(c=>String(c.id)===String(chatId));
    if(local){local.messages=d.chat.messages||[];local.title=d.chat.title||local.title;}
    renderSidebar(); renderMessages(d.chat.messages||[]); cancel();

    // Submit exactly one new question: the edited version.
    const input=$('questionInput');
    input.value=edited;
    autoResize();
    await sendQuestion();
  };

  box.querySelector('.question-edit-cancel').onclick=cancel;
  box.querySelector('.question-edit-send').onclick=submit;
  textarea.addEventListener('keydown',e=>{
    if(e.key==='Escape'){e.preventDefault();cancel();}
    if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();submit();}
  });
  requestAnimationFrame(()=>{textarea.focus();textarea.select();});
}

function uniqueSourceLabels(sources){
  const seen=new Set(),out=[];
  (sources||[]).forEach(x=>{
    const name=typeof x==='string'?x:(x.title||x.name||x.filename||x.material_name||x.url||'');
    const page=(typeof x==='object'&&x&&x.page)?` — Page ${x.page}`:'';
    const label=(name+page).trim();
    if(label&&!seen.has(label)){seen.add(label);out.push(label);}
  });
  return out;
}

function formatTableCell(cell){
  let v=cell.trim();
  v=v.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>');
  v=v.replace(/`([^`]+)`/g,'<code class="inline-code">$1</code>');
  return v;
}

function formatMarkdownTables(s){
  const lines=s.split('\n'),out=[]; let i=0;
  const separator=/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/;
  while(i<lines.length){
    const line=lines[i], next=lines[i+1]||'';
    if(line.includes('|')&&separator.test(next)){
      const headers=line.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(formatTableCell);
      i+=2; const rows=[];
      while(i<lines.length&&lines[i].includes('|')&&lines[i].trim()!==''){
        rows.push(lines[i].trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(formatTableCell)); i++;
      }
      let html='<div class="table-scroll"><table class="answer-table"><thead><tr>';
      headers.forEach(c=>html+=`<th>${c}</th>`); html+='</tr></thead><tbody>';
      rows.forEach(cells=>{html+='<tr>'; for(let j=0;j<headers.length;j++) html+=`<td>${cells[j]||''}</td>`; html+='</tr>';});
      html+='</tbody></table></div>'; out.push(html); continue;
    }
    out.push(line); i++;
  }
  return out.join('\n');
}

function formatAnswer(text){
  let s=escapeHtml(String(text));
  s=formatMarkdownTables(s);
  s=s.replace(/```([\s\S]*?)```/g,'<pre><code>$1</code></pre>');
  s=s.replace(/^### (.*)$/gm,'<h4>$1</h4>').replace(/^## (.*)$/gm,'<h3>$1</h3>').replace(/^# (.*)$/gm,'<h2>$1</h2>');
  s=s.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/`([^`]+)`/g,'<code class="inline-code">$1</code>');
  s=s.replace(/^\s*[-•] (.*)$/gm,'<li>$1</li>');
  s=s.replace(/(<li>[\s\S]*?<\/li>)/g,'<ul>$1</ul>');
  s=s.replace(/^(\d+\. )(.+)$/gm,'<div class="numbered"><strong>$1</strong>$2</div>');
  s=s.replace(/\n/g,'<br>');
  s=s.replace(/<\/ul><br><ul>/g,'</ul><ul>');
  return s;
}

function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

function setupComposerDropzone(){
  const composer=$('composer');
  const input=$('questionInput');
  ['dragenter','dragover'].forEach(type=>{
    composer.addEventListener(type,e=>{
      e.preventDefault();
      e.stopPropagation();
      composer.classList.add('drag-over');
    });
  });
  ['dragleave','drop'].forEach(type=>{
    composer.addEventListener(type,e=>{
      e.preventDefault();
      e.stopPropagation();
      if(type==='dragleave' && e.relatedTarget && composer.contains(e.relatedTarget)) return;
      composer.classList.remove('drag-over');
    });
  });
  composer.addEventListener('drop',e=>{
    const files=Array.from(e.dataTransfer.files||[]);
    if(!files.length) return;
    const dt=new DataTransfer();
    Array.from($('fileInput').files||[]).forEach(f=>dt.items.add(f));
    files.forEach(f=>dt.items.add(f));
    $('fileInput').files=dt.files;
    state.attachments=Array.from(dt.files);
    renderAttachments();
    showToast(`${files.length} file${files.length===1?'':'s'} attached.`);
    input.focus();
  });
}
function setupComposerViewport(){
  const wrap=$('composer').closest('.composer-wrap');
  const scroll=$('chatScroll');
  if(!wrap||!scroll)return;
  const update=()=>{
    const h=wrap.getBoundingClientRect().height;
    document.documentElement.style.setProperty('--composer-space',`${Math.ceil(h)+8}px`);
    scroll.style.paddingBottom = `${Math.max(24, Math.ceil(h)+18)}px`;
  };
  if(window.ResizeObserver){ const ro=new ResizeObserver(update); ro.observe(wrap); }
  window.addEventListener('resize',update,{passive:true});
  if(window.visualViewport) window.visualViewport.addEventListener('resize',update,{passive:true});
  requestAnimationFrame(update);
}

async function sendQuestion(){
  const input=$('questionInput'),question=input.value.trim();
  if(!question || getActiveRunningJob())return;
  if(!state.activeChatId){await createNewChat();}
  const active=state.chats.find(c=>String(c.id)===String(state.activeChatId));
  if(active){
    active.messages=active.messages||[];
    active.messages.push({role:'user',content:question});
    active.title=makeTitle(question);
    renderSidebar();
  }

  const userIndex = active ? ((active.messages||[]).length - 1) : 0;
  appendMessage('user',question,[],active?.messages?.[userIndex]?.id||'',userIndex,active?.messages||[]);
  showThinking();
  scrollBottom();
  input.value='';
  autoResize();

  const form=new FormData();
  form.append('question',question);
  form.append('chat_id',state.activeChatId);
  form.append('learning_mode',state.learningMode?'1':'0');
  state.attachments.forEach(f=>form.append('files',f));
  state.attachments=[];
  $('fileInput').value='';
  renderAttachments();
  setGenerating(true);

  try{
    const r=await fetch('/api/chat',{method:'POST',body:form});
    const d=await r.json();
    if(!d.ok){
      removeThinking();
      setGenerating(false);
      appendMessage('assistant',d.error||'Something went wrong.');
      return;
    }
    state.backgroundJobs[d.job_id]={
      jobId:d.job_id,
      chatId:state.activeChatId,
      status:'running'
    };
    state.pollingJobId=d.job_id;
    pollJob(d.job_id,state.activeChatId);
  }catch(_){
    removeThinking();
    setGenerating(false);
    appendMessage('assistant','Connection error. Please try again.');
  }
}

function pollJob(jobId,chatId){
  const job=state.backgroundJobs[jobId];
  if(!job)return;

  clearInterval(job.timer);
  job.timer=setInterval(async()=>{
    try{
      const r=await fetch(`/api/jobs/${jobId}`);
      const d=await r.json();

      if(!d.ok){
        clearInterval(job.timer);
        delete state.backgroundJobs[jobId];
        if(String(state.activeChatId)===String(chatId)){
          state.pollingJobId=null;
          removeThinking();
          setGenerating(false);
        }
        return;
      }

      if(d.status==='done'){
        clearInterval(job.timer);
        delete state.backgroundJobs[jobId];

        // The backend has already saved the authoritative answer to the
        // original chat. Reload that chat instead of constructing a second
        // local copy. This also makes chat switching safe.
        let target=state.chats.find(c=>String(c.id)===String(chatId));
        try{
          const cr=await fetch(`/api/chats/${chatId}`);
          if(cr.ok){
            const cd=await cr.json();
            if(cd.ok && cd.chat){
              if(target){
                target.messages=cd.chat.messages||[];
                target.title=cd.chat.title||target.title;
              }else{
                target={...cd.chat};
                state.chats.push(target);
              }
            }
          }
        }catch(_){}

        try{ await fetch(`/api/jobs/${jobId}/ack`,{method:'POST'}); }catch(_){}

        if(String(state.activeChatId)===String(chatId)){
          state.pollingJobId=null;
          removeThinking();
          setGenerating(false);
          if(target){
            renderMessages(target.messages||[]);
            scrollBottom(true);
          }
        }else{
          renderSidebar();
          showBackgroundCompletionNotification(
            chatId,
            d.answer || target?.messages?.at(-1)?.content || ''
          );
        }
        return;
      }

      if(d.status==='error'){
        clearInterval(job.timer);
        delete state.backgroundJobs[jobId];

        try{ await fetch(`/api/jobs/${jobId}/ack`,{method:'POST'}); }catch(_){}

        if(String(state.activeChatId)===String(chatId)){
          state.pollingJobId=null;
          removeThinking();
          setGenerating(false);
          appendMessage('assistant',d.answer||'Generation failed. Please try again.');
          if(d.detail) console.error('AI-LMS generation error:',d.detail);
          showToast('AI generation failed. Check the server log if it repeats.');
        }else{
          renderSidebar();
        }
        return;
      }

      if(d.status==='cancelled'){
        clearInterval(job.timer);
        delete state.backgroundJobs[jobId];
        if(String(state.activeChatId)===String(chatId)){
          state.pollingJobId=null;
          removeThinking();
          setGenerating(false);
        }
      }
    }catch(e){
      clearInterval(job.timer);
      delete state.backgroundJobs[jobId];
      if(String(state.activeChatId)===String(chatId)){
        state.pollingJobId=null;
        removeThinking();
        setGenerating(false);
        appendMessage('assistant','Connection error while checking the generation. Please try again.');
      }
    }
  },400);
}

async function stopGeneration(){
  const activeJob=getActiveRunningJob();
  if(!activeJob)return;

  clearInterval(activeJob.timer);
  delete state.backgroundJobs[activeJob.jobId];

  if(String(state.pollingJobId)===String(activeJob.jobId)) state.pollingJobId=null;
  removeThinking();
  setGenerating(false);

  try{
    await fetch(`/api/jobs/${activeJob.jobId}/stop`,{method:'POST'});
  }catch(_){}

  showToast('Generation stopped.');
}

function setGenerating(active){
  const b=$('sendBtn');
  b.classList.toggle('stop',!!active);
  b.classList.toggle('ready',!active&&$('questionInput').value.trim().length>0);
  b.title=active?'Stop generation':'Send';
}
function updateSendButton(){
  const active=!!getActiveRunningJob();
  $('sendBtn').classList.toggle('stop',active);
  $('sendBtn').classList.toggle('ready',!active&&$('questionInput').value.trim().length>0);
  $('sendBtn').title=active?'Stop generation':'Send';
}
function autoResize(){const i=$('questionInput');i.style.height='auto';i.style.height=Math.min(i.scrollHeight,150)+'px';updateSendButton();}
function renderAttachments(){const b=$('attachmentList');b.innerHTML='';state.attachments.forEach(f=>{const c=document.createElement('div');c.className='attachment-chip';c.textContent=f.name;b.appendChild(c);});}
function scrollBottom(force=true){requestAnimationFrame(()=>{$('chatScroll').scrollTo({top:$('chatScroll').scrollHeight,behavior:force?'smooth':'auto'});updateJumpButton();});}
function updateJumpButton(){const s=$('chatScroll'),away=s.scrollHeight-s.scrollTop-s.clientHeight>180;$('jumpLatest').classList.toggle('show',away);}
function makeTitle(t){const c=t.replace(/\s+/g,' ').trim();return c.length>42?c.slice(0,42).trimEnd()+'...':c;}
function toggleSidebar(){
  if (window.innerWidth <= 800) {
    if (document.body.classList.contains('sidebar-open')) closeSidebar();
    else openSidebar();
  } else {
    document.body.classList.toggle('sidebar-collapsed');
  }
  setTimeout(()=>updateJumpButton(),220);
}
function openSidebar(){document.body.classList.remove('sidebar-collapsed');document.body.classList.add('sidebar-open');$('sidebarToggle').setAttribute('aria-label','Hide sidebar');$('sidebarToggle').setAttribute('title','Hide sidebar');}
function closeSidebar(){document.body.classList.remove('sidebar-open');document.body.classList.add('sidebar-collapsed');$('sidebarToggle').setAttribute('aria-label','Show sidebar');$('sidebarToggle').setAttribute('title','Show sidebar');}
function toggleTheme(){state.dark=!state.dark;localStorage.setItem('ai-lms-theme',state.dark?'dark':'light');applyTheme();}
function applyTheme(){document.body.classList.toggle('dark',state.dark);$('themeBtn').textContent=state.dark?'☾':'☼';}
async function copyAnswer(text){try{await navigator.clipboard.writeText(text);showToast('Answer copied.');}catch(_){showToast('Copy was not available.');}}
async function retryAnswer(messageId){
  if(state.polling)return;
  const id=state.activeChatId;
  if(!id)return;
  setGenerating(true);
  const r=await fetch(`/api/chats/${id}/retry`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message_id:messageId})});
  const d=await r.json();
  if(!d.ok){setGenerating(false);showToast(d.error||'Could not retry.');return;}
  const local=state.chats.find(c=>String(c.id)===String(id));
  if(local && local.messages){
    const ix=local.messages.findIndex(m=>String(m.id||'')===String(messageId));
    if(ix>=0) local.messages.splice(ix,1);
    else if(local.messages[local.messages.length-1]?.role==='assistant') local.messages.pop();
  }
  renderMessages(local?.messages||[]);
  showThinking();
  scrollBottom();
  state.backgroundJobs[d.job_id]={jobId:d.job_id,chatId:id,status:'running'};
  state.pollingJobId=d.job_id;
  pollJob(d.job_id,id);
}
async function downloadAnswerPdf(chatId,messageId){showToast('Preparing PDF...');const r=await fetch(`/api/chats/${chatId}/answer-pdf`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message_id:messageId})});if(!r.ok){const d=await r.json().catch(()=>({}));showToast(d.error||'Could not create PDF.');return;}const blob=await r.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='AI-LMS-Answer.pdf';a.click();URL.revokeObjectURL(url);showToast('PDF generated.');}
function showToast(text){const t=$('toast');t.textContent=text;t.classList.add('show');clearTimeout(showToast.timer);showToast.timer=setTimeout(()=>t.classList.remove('show'),2200);}

function showBackgroundCompletionNotification(chatId, answer){
  const chat=state.chats.find(c=>String(c.id)===String(chatId));
  const title=chat?.title || 'AI Tutor';
  const preview=String(answer || '').replace(/\s+/g,' ').trim();

  const notification={
    id:'notification-'+Date.now()+'-'+Math.random().toString(36).slice(2),
    chatId:String(chatId),
    title:title,
    preview:preview ? (preview.length>90 ? preview.slice(0,90)+'…' : preview) : 'Your answer is ready.'
  };

  state.backgroundNotifications.unshift(notification);
  state.backgroundNotifications=state.backgroundNotifications.slice(0,5);
  renderBackgroundNotifications();
}

function renderBackgroundNotifications(){
  let container=document.getElementById('backgroundNotifications');

  if(!container){
    container=document.createElement('div');
    container.id='backgroundNotifications';
    container.className='background-notifications';
    document.body.appendChild(container);
  }

  container.innerHTML='';

  state.backgroundNotifications.forEach(notification=>{
    const item=document.createElement('div');
    item.className='background-notification';
    item.innerHTML=`
      <div class="background-notification-icon">✓</div>
      <div class="background-notification-content">
        <div class="background-notification-title">Answer ready</div>
        <div class="background-notification-chat">${escapeHtml(notification.title)}</div>
        <div class="background-notification-preview">${escapeHtml(notification.preview)}</div>
      </div>
      <button type="button" class="background-notification-close" aria-label="Dismiss notification">×</button>
    `;

    item.addEventListener('click',async e=>{
      if(e.target.closest('.background-notification-close')){
        removeBackgroundNotification(notification.id);
        return;
      }
      removeBackgroundNotification(notification.id);
      await selectChat(notification.chatId);
    });

    container.appendChild(item);
  });
}

function removeBackgroundNotification(notificationId){
  state.backgroundNotifications=state.backgroundNotifications.filter(n=>n.id!==notificationId);
  renderBackgroundNotifications();
}




/* =========================================================
   PREMIUM AI THINKING ANIMATION
   Appears as an assistant-side element directly below the
   student's question. It is removed when generation ends.
   ========================================================= */
let thinkingTimer = null;
const thinkingMessages = [
  'Understanding your question',
  'Searching your course material',
  'Analyzing relevant concepts',
  'Preparing a clear explanation'
];
let thinkingIndex = 0;

function showThinking(){
  removeThinking(true);
  const messages=$('messages');
  if(!messages)return;

  const wrapper=document.createElement('article');
  wrapper.id='aiThinking';
  wrapper.className='message assistant thinking-message';
  wrapper.innerHTML=`
    <div class="thinking-avatar" aria-hidden="true">
      <div class="thinking-avatar-glow"></div>
      <div class="thinking-avatar-core">AI</div>
    </div>
    <div class="thinking-card">
      <div class="thinking-animation" aria-hidden="true">
        <span class="thinking-dot dot-1"></span>
        <span class="thinking-dot dot-2"></span>
        <span class="thinking-dot dot-3"></span>
        <span class="thinking-dot dot-4"></span>
      </div>
      <div class="thinking-title">AI is thinking<span class="thinking-ellipsis">...</span></div>
      <div class="thinking-status" id="thinkingStatus">${thinkingMessages[0]}</div>
      <div class="thinking-line" aria-hidden="true">
        <span></span><span></span><span></span><span></span><span></span><span></span><span></span>
      </div>
    </div>`;

  messages.appendChild(wrapper);
  thinkingIndex=0;

  thinkingTimer=setInterval(()=>{
    thinkingIndex=(thinkingIndex+1)%thinkingMessages.length;
    const status=$('thinkingStatus');
    if(!status)return;
    status.style.opacity='0';
    status.style.transform='translateY(4px)';
    setTimeout(()=>{
      const current=$('thinkingStatus');
      if(!current)return;
      current.textContent=thinkingMessages[thinkingIndex];
      current.style.opacity='1';
      current.style.transform='translateY(0)';
    },220);
  },2200);

  requestAnimationFrame(()=>{
    const scroll=$('chatScroll');
    if(scroll){
      const thinking=document.getElementById('aiThinking');
      if(thinking) thinking.scrollIntoView({behavior:'smooth',block:'nearest'});
      else scroll.scrollTo({top:scroll.scrollHeight,behavior:'smooth'});
    }
  });
}

function removeThinking(immediate=false){
  if(thinkingTimer){
    clearInterval(thinkingTimer);
    thinkingTimer=null;
  }
  const thinking=$('aiThinking');
  if(!thinking)return;
  if(immediate){thinking.remove();return;}
  thinking.classList.add('thinking-leaving');
  setTimeout(()=>{
    const current=$('aiThinking');
    if(current)current.remove();
  },180);
}

/* =========================================================
   CONTEXT-AWARE LEARNING MODE + YOUTUBE
   ========================================================= */
function wireLearningMode(){
  const btn=$('learningModeBtn');
  if(!btn)return;
  btn.addEventListener('click',()=>{
    state.learningMode=!state.learningMode;
    localStorage.setItem('ai-lms-learning-mode',state.learningMode?'1':'0');
    applyLearningMode();
    showToast(state.learningMode?'Learning Mode enabled.':'Learning Mode disabled.');
  });
  applyLearningMode();
}
function applyLearningMode(){
  const btn=$('learningModeBtn');
  if(!btn)return;
  btn.classList.toggle('active',state.learningMode);
  const label=state.learningMode?'Learning Mode: On':'Learning Mode: Off';
  btn.title=label;
  btn.setAttribute('aria-label',label);
}

function getHistoryText(){
  const active=state.chats.find(c=>String(c.id)===String(state.activeChatId));
  if(!active)return '';
  return (active.messages||[]).slice(-8).map(m=>`${m.role==='user'?'Student':'AI Tutor'}: ${m.content||''}`).join('\n');
}

async function getYoutubeRecommendation(question,answer,history){
  try{
    const r=await fetch('/api/youtube-recommendation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,answer,history})});
    const d=await r.json();
    return d.ok?d.youtube:null;
  }catch(_){return null;}
}

async function toggleYoutubeForMessage(anchor,messageId,answer,existing,learningMode){
  closeYoutubePopover();
  let video=existing;
  if(!video){
    const active=state.chats.find(c=>String(c.id)===String(state.activeChatId));
    const msgs=active?.messages||[];
    let question='';
    for(let i=msgs.length-1;i>=0;i--){
      if(msgs[i].role==='assistant' && String(msgs[i].id||'')===String(messageId)){
        for(let j=i-1;j>=0;j--){if(msgs[j].role==='user'){question=msgs[j].content||'';break;}}
        break;
      }
    }
    if(!question){for(let i=msgs.length-1;i>=0;i--){if(msgs[i].role==='user'){question=msgs[i].content||'';break;}}}
    video=await getYoutubeRecommendation(question,answer,getHistoryText());
    if(video){
      const m=msgs.find(x=>String(x.id||'')===String(messageId));
      if(m)m.youtube=video;
    }
  }
  showYoutubePopover(anchor,video);
}

function showYoutubePopover(anchor,video){
  const pop=$('youtubePopover'); if(!pop)return;
  if(!video){
    pop.innerHTML='<div class="youtube-popover-head"><span class="youtube-play-mini">'+actionIcon('youtube')+'</span><div><strong>YouTube recommendation</strong><small>No direct video was found</small></div><button class="youtube-close" aria-label="Close">×</button></div><div class="youtube-empty">No suitable educational video was found for this topic.</div><button class="youtube-search" type="button">Search this topic on YouTube</button>';
    pop.querySelector('.youtube-search').onclick=()=>{const active=state.chats.find(c=>String(c.id)===String(state.activeChatId));const msgs=active?.messages||[];let q='';for(let i=msgs.length-1;i>=0;i--){if(msgs[i].role==='user'){q=msgs[i].content||'';break;}}window.open('https://www.youtube.com/results?search_query='+encodeURIComponent(q),'_blank','noopener,noreferrer');};
  }else{
    const title=escapeHtml(video.title||'Recommended video'),channel=escapeHtml(video.channel||'YouTube'),url=String(video.url||'');
    pop.innerHTML=`<div class="youtube-popover-head"><span class="youtube-play-mini">${actionIcon('youtube')}</span><div><strong>Recommended video</strong><small>${channel}</small></div><button class="youtube-close" aria-label="Close">×</button></div><div class="youtube-video-title">${title}</div><a class="youtube-watch" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Watch on YouTube</a>`;
  }
  pop.querySelector('.youtube-close').onclick=closeYoutubePopover;
  pop.classList.add('open'); pop.setAttribute('aria-hidden','false');
  const r=anchor.getBoundingClientRect(),width=Math.min(350,window.innerWidth-20);
  pop.style.width=width+'px'; pop.style.left='10px'; pop.style.top='10px';
  const h=pop.offsetHeight; let left=r.right-width;
  if(left<10)left=10; if(left+width>window.innerWidth-10)left=window.innerWidth-width-10;
  let top=r.bottom+8; if(top+h>window.innerHeight-10)top=r.top-h-8; if(top<10)top=10;
  pop.style.left=left+'px'; pop.style.top=top+'px';
}
function closeYoutubePopover(){const p=$('youtubePopover');if(p){p.classList.remove('open');p.setAttribute('aria-hidden','true');p.innerHTML='';}}

function showSourcesPopover(anchor,sources){
  closeYoutubePopover(); const pop=$('sourcesPopover'); if(!pop)return;
  const unique=[],seen=new Set();
  (sources||[]).forEach(x=>{if(!x)return;const name=typeof x==='string'?x:(x.title||x.name||x.filename||x.material_name||'Course material');const page=typeof x==='object'&&x&&x.page?x.page:null;const key=name+'|'+(page||'');if(!seen.has(key)){seen.add(key);unique.push({name,page,url:typeof x==='object'?x.url:null});}});
  pop.innerHTML='<div class="sources-popover-head"><span class="sources-book">'+actionIcon('sources')+'</span><div><strong>Course Material Sources</strong><small>The material pages used for this answer</small></div><button class="sources-close" aria-label="Close">×</button></div><div class="sources-list">'+unique.map(x=>`<div class="source-item"><strong>${escapeHtml(x.name)}</strong>${x.page?`<span>Page ${escapeHtml(x.page)}</span>`:''}${x.url?`<a href="${escapeHtml(x.url)}" target="_blank" rel="noopener noreferrer">Open source</a>`:''}</div>`).join('')+'</div>';
  pop.querySelector('.sources-close').onclick=closeSourcesPopover;
  pop.classList.add('open');pop.setAttribute('aria-hidden','false');
  const r=anchor.getBoundingClientRect(),width=Math.min(380,window.innerWidth-20);pop.style.width=width+'px';pop.style.left=Math.max(10,Math.min(window.innerWidth-width-10,r.left-width+40))+'px';pop.style.top='10px';const h=pop.offsetHeight;let top=r.bottom+8;if(top+h>window.innerHeight-10)top=r.top-h-8;if(top<10)top=10;pop.style.top=top+'px';
}
function closeSourcesPopover(){const p=$('sourcesPopover');if(p){p.classList.remove('open');p.setAttribute('aria-hidden','true');p.innerHTML='';}}
window.addEventListener('resize',()=>{closeYoutubePopover();closeSourcesPopover();},{passive:true});
window.addEventListener('click',e=>{const yp=$('youtubePopover'),sp=$('sourcesPopover');if(yp?.classList.contains('open')&&!yp.contains(e.target)&&!e.target.closest('[data-action="youtube"]'))closeYoutubePopover();if(sp?.classList.contains('open')&&!sp.contains(e.target)&&!e.target.closest('[data-action="sources"]'))closeSourcesPopover();});