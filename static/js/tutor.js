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
  $('chatMenuMove').addEventListener('click', menuMove);
  $('folderMenuRename').addEventListener('click', menuRenameFolder);
  $('folderMenuPin').addEventListener('click', menuPinFolder);
  $('folderMenuDelete').addEventListener('click', menuDeleteFolder);
  document.addEventListener('click', e => {
    if (!$('profileMenu').contains(e.target) && e.target !== $('profileMenuBtn')) $('profileMenu').classList.remove('open');
    if (!$('chatMenu').contains(e.target) && !e.target.closest('.more-action')) closeChatMenu();
    if (!$('folderMenu').contains(e.target) && !e.target.closest('.folder-more-action')) closeFolderMenu();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeChatMenu(); closeFolderMenu(); closeDeleteModal(); closeRenameModal(); closeFolderModal(); closeMoveModal(); closeYoutubePopover(); } });
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
  toggle.innerHTML=window.leeLearnIcon(expanded?'chevronDown':'chevronRight'); toggle.setAttribute('aria-label',expanded?'Collapse folder':'Expand folder');
  const icon=document.createElement('span'); icon.className='folder-icon'; icon.innerHTML=window.leeLearnIcon('folder');
  const name=document.createElement('span'); name.className='folder-name'; name.textContent=folder.name;
  const newChat=document.createElement('button'); newChat.type='button'; newChat.className='folder-header-action folder-new-chat-action';
  newChat.innerHTML=window.leeLearnIcon('squarePen'); newChat.title='New chat in folder'; newChat.setAttribute('aria-label',`New chat in ${folder.name}`);
  newChat.addEventListener('click',e=>{e.stopPropagation();createChatInFolder(folder);});
  const more=document.createElement('button'); more.type='button'; more.className='folder-header-action folder-more-action';
  more.innerHTML=window.leeLearnIcon('moreHorizontal'); more.title='Folder options'; more.setAttribute('aria-label',`Options for ${folder.name}`);
  more.addEventListener('click',e=>{e.stopPropagation();openFolderMenu(folder,more);});
  head.addEventListener('click',()=>{state.expandedFolders[String(folder.id)]=!expanded;localStorage.setItem('ai-lms-expanded-folders',JSON.stringify(state.expandedFolders));renderSidebar();});
  head.append(toggle,icon,name,newChat,more); head.addEventListener('dragover',e=>{e.preventDefault(); head.classList.add('drag-over');}); head.addEventListener('dragleave',()=>head.classList.remove('drag-over')); head.addEventListener('drop',async e=>{e.preventDefault(); head.classList.remove('drag-over'); const id=e.dataTransfer.getData('text/chat-id')||state.draggedChatId; if(id) await moveChat(id,folder.id);});
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

  if(!chat.folder_id){
    const folderBtn=document.createElement('button');
    folderBtn.type='button';
    folderBtn.className='chat-action folder-action';
    folderBtn.innerHTML=window.leeLearnIcon('folder');
    folderBtn.title='Move to folder';
    folderBtn.setAttribute('aria-label','Move to folder');
    folderBtn.addEventListener('click',e=>{
      e.stopPropagation();
      openMoveModal(chat.id);
    });
    actions.appendChild(folderBtn);
  }

  // Pin remains available in chat options instead of adding a row button.
  const more=document.createElement('button');
  more.type='button';
  more.className='chat-action more-action';
  more.innerHTML=window.leeLearnIcon('moreHorizontal');
  more.title='Chat options';
  more.setAttribute('aria-label','Chat options');
  more.addEventListener('click',e=>{
    e.stopPropagation();
    openChatMenu(chat,more);
  });

  actions.append(more);
  row.append(title,actions);
  requestAnimationFrame(()=>{
    if(titleText.scrollWidth > title.clientWidth + 4) title.classList.add('overflow');
  });
  return row;
}
let menuChatId=null, menuFolderId=null, pendingDeleteChatId=null, renameChatId=null, moveChatId=null;
function positionPopupMenu(menu,anchor,height){const r=anchor.getBoundingClientRect(),w=190;let left=r.right-w,top=r.bottom+4;if(left<8)left=8;if(left+w>innerWidth-8)left=innerWidth-w-8;if(top+height>innerHeight-8)top=r.top-height-4;if(top<8)top=8;menu.style.left=left+'px';menu.style.top=top+'px';}
function openChatMenu(chat,anchor){closeFolderMenu();closeChatMenu();menuChatId=chat.id;const m=$('chatMenu');$('chatMenuPin').querySelector('.ui-icon-label').textContent=chat.pinned?'Unpin chat':'Pin chat';positionPopupMenu(m,anchor,170);m.classList.add('open');}
function closeChatMenu(){$('chatMenu').classList.remove('open');menuChatId=null;}
function openFolderMenu(folder,anchor){closeChatMenu();closeFolderMenu();menuFolderId=folder.id;const menu=$('folderMenu');$('folderMenuPin').querySelector('.ui-icon-label').textContent=folder.pinned?'Unpin folder':'Pin folder';positionPopupMenu(menu,anchor,130);menu.classList.add('open');}
function closeFolderMenu(){$('folderMenu').classList.remove('open');menuFolderId=null;}
async function menuPin(){const c=state.chats.find(x=>String(x.id)===String(menuChatId));closeChatMenu();if(c)await togglePinChat(c);}
function menuRename(){const c=state.chats.find(x=>String(x.id)===String(menuChatId));closeChatMenu();if(c)openRenameModal(c);}
function menuDelete(){pendingDeleteChatId=menuChatId;closeChatMenu();if(pendingDeleteChatId)openDeleteModal();}
function menuMove(){const id=menuChatId;closeChatMenu();if(id)openMoveModal(id);}
async function togglePinChat(chat){const r=await fetch(`/api/chats/${chat.id}/pin`,{method:'POST'});const d=await r.json();if(d.ok){chat.pinned=!!d.pinned;renderSidebar();}}
async function menuRenameFolder(){const folder=state.folders.find(item=>String(item.id)===String(menuFolderId));closeFolderMenu();if(!folder)return;const name=window.prompt('Rename folder',folder.name);if(name===null)return;const cleaned=name.trim();if(!cleaned)return;const response=await fetch(`/api/folders/${folder.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:cleaned})});const data=await response.json();if(!response.ok||!data.ok){showToast(data.error||'Could not rename folder.');return;}folder.name=data.folder.name;renderSidebar();}
async function menuPinFolder(){const folder=state.folders.find(item=>String(item.id)===String(menuFolderId));closeFolderMenu();if(!folder)return;const response=await fetch(`/api/folders/${folder.id}/pin`,{method:'POST'});const data=await response.json();if(!response.ok||!data.ok){showToast(data.error||'Could not update folder pin.');return;}Object.assign(folder,data.folder);renderSidebar();}
async function menuDeleteFolder(){const folder=state.folders.find(item=>String(item.id)===String(menuFolderId));closeFolderMenu();if(!folder)return;if(!window.confirm(`Delete folder "${folder.name}"? Its chats will be moved to Recent Chats.`))return;const response=await fetch(`/api/folders/${folder.id}`,{method:'DELETE'});const data=await response.json();if(!response.ok||!data.ok){showToast(data.error||'Could not delete folder.');return;}state.folders=state.folders.filter(item=>String(item.id)!==String(folder.id));state.chats.forEach(chat=>{if(String(chat.folder_id||'')===String(folder.id))chat.folder_id=null;});delete state.expandedFolders[String(folder.id)];localStorage.setItem('ai-lms-expanded-folders',JSON.stringify(state.expandedFolders));renderSidebar();}

function openDeleteModal(){ $('chatActionModal').classList.add('open'); $('chatDeleteError').textContent=''; }
function closeDeleteModal(){ $('chatActionModal').classList.remove('open'); pendingDeleteChatId=null; }
async function confirmChatDelete(){const id=pendingDeleteChatId;if(!id)return;const r=await fetch(`/api/chats/${id}`,{method:'DELETE'});const d=await r.json();if(!d.ok){$('chatDeleteError').textContent=d.error||'Could not delete chat.';return;}state.chats=state.chats.filter(c=>String(c.id)!==String(id));closeDeleteModal();if(String(state.activeChatId)===String(id)){state.activeChatId=state.chats[0]?.id||null;if(state.activeChatId)await selectChat(state.activeChatId);else{renderMessages([]);showWelcome();}}renderSidebar();}

function openRenameModal(chat){renameChatId=chat.id;$('renameInput').value=chat.title||'';$('renameError').textContent='';$('renameModal').classList.add('open');requestAnimationFrame(()=>{$('renameInput').focus();$('renameInput').select();});}
function closeRenameModal(){$('renameModal').classList.remove('open');renameChatId=null;}
async function saveChatRename(){const id=renameChatId,title=$('renameInput').value.trim();if(!id||!title)return;const r=await fetch(`/api/chats/${id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title})});const d=await r.json();if(!d.ok){$('renameError').textContent=d.error||'Could not rename chat.';return;}const c=state.chats.find(x=>String(x.id)===String(id));if(c)c.title=d.chat.title;closeRenameModal();renderSidebar();}

function openFolderModal(){$('folderInput').value='';$('folderError').textContent='';$('folderModal').classList.add('open');requestAnimationFrame(()=>$('folderInput').focus());}
function closeFolderModal(){$('folderModal').classList.remove('open');}
async function saveFolder(){const name=$('folderInput').value.trim();if(!name)return;const r=await fetch('/api/folders',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});const d=await r.json();if(!d.ok){$('folderError').textContent=d.error||'Could not create folder.';return;}state.folders.unshift(d.folder);state.expandedFolders[String(d.folder.id)]=true;closeFolderModal();renderSidebar();}

function openMoveModal(chatId){moveChatId=chatId;const box=$('folderChoices');box.innerHTML='';const none=document.createElement('button');none.className='folder-choice';none.innerHTML=`<span class="ui-icon-slot">${window.leeLearnIcon('folder')}</span><span>Remove from folder</span>`;none.onclick=()=>moveChat(chatId,null);box.appendChild(none);if(!state.folders.length){box.appendChild(emptyText('No folders created yet.'));}else state.folders.forEach(f=>{const b=document.createElement('button');b.className='folder-choice';b.innerHTML=`<span class="ui-icon-slot">${window.leeLearnIcon('folder')}</span><span>${escapeHtml(f.name)}</span><span class="choice-count">${state.chats.filter(c=>String(c.folder_id)===String(f.id)).length}</span>`;b.onclick=()=>moveChat(chatId,f.id);box.appendChild(b);});$('moveModal').classList.add('open');}
function closeMoveModal(){$('moveModal').classList.remove('open');moveChatId=null;}
async function moveChat(chatId,folderId){const r=await fetch(`/api/chats/${chatId}/move`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder_id:folderId})});const d=await r.json();if(!d.ok){showToast(d.error||'Could not move chat.');return;}const c=state.chats.find(x=>String(x.id)===String(chatId));if(c)c.folder_id=folderId;closeMoveModal();renderSidebar();showToast(folderId?'Chat moved to folder.':'Chat removed from folder.');}

async function createNewChat(){const r=await fetch('/api/chats',{method:'POST'});const d=await r.json();if(!d.ok){showToast(d.error||'Could not create chat.');return;}state.chats.unshift(d.chat);state.activeChatId=d.chat.id;renderSidebar();renderMessages([]);showWelcome();$('questionInput').focus();}
async function createChatInFolder(folder){const response=await fetch(`/api/folders/${folder.id}/chats`,{method:'POST'});const data=await response.json();if(!response.ok||!data.ok){showToast(data.error||'Could not create chat in folder.');return;}state.chats.unshift(data.chat);state.activeChatId=data.chat.id;state.expandedFolders[String(folder.id)]=true;localStorage.setItem('ai-lms-expanded-folders',JSON.stringify(state.expandedFolders));$('chatSearch').value='';renderSidebar();renderMessages(data.chat.messages||[]);$('questionInput').focus();}
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
  messages.forEach((m,i)=>appendMessage(m.role,m.content,m.sources||[],m.id||'',i,messages,m.youtube||null,m.learning_mode||false,m.diagram||null));
  const running=getActiveRunningJob();
  if(running) showThinking();
  scrollBottom(false);
}
function actionIcon(name){
  return window.leeLearnIcon(name==='edit'?'pencil':name);
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

function appendMessage(role,content,sources=[],messageId='',index=0,messages=[],youtube=null,learningMode=false,diagram=null){
  hideWelcome();
  const wrap=document.createElement('article');
  wrap.className=`message ${role}`;
  if(messageId)wrap.dataset.messageId=messageId;
  wrap.dataset.messageIndex=String(index);

  const body=document.createElement('div'); body.className='message-body';
  const bubble=document.createElement('div'); bubble.className='message-bubble';

  if(role==='assistant'){
    try{
      const rendered=formatAnswer(content||'');
      if(rendered)bubble.innerHTML=rendered;
      else bubble.textContent=content||'';
    }catch(_){
      bubble.textContent=content||'';
    }
    body.appendChild(bubble);
    appendGroundedDiagram(body,diagram);
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
  if(role==='assistant')typesetAnswerMath(bubble);
}

let groundedDiagramSequence=0;
function appendGroundedDiagram(body,diagram){
  if(!diagram||diagram.grounded!==true||diagram.derived!==true||!diagram.mermaid)return;
  const section=document.createElement('section');
  section.className='grounded-diagram';
  const label=document.createElement('div');
  label.className='grounded-diagram-label';
  label.textContent='Diagram derived from the course material';
  const visual=document.createElement('div');
  visual.className='grounded-diagram-visual';
  section.append(label,visual);
  body.appendChild(section);
  renderGroundedMermaid(visual,diagram.mermaid);
}

function renderGroundedMermaid(container,source){
  const showSource=()=>{
    container.replaceChildren();
    const pre=document.createElement('pre');
    pre.className='answer-code-block';
    const code=document.createElement('code');
    code.className='language-mermaid';
    code.textContent=source;
    pre.appendChild(code);
    container.appendChild(pre);
  };

  if(!window.mermaid||typeof window.mermaid.render!=='function'){
    showSource();
    return;
  }

  const diagramSource=String(source||'').trim();
  if(!/^(?:flowchart|graph)\s+(?:TD|TB|BT|LR|RL)\b/.test(diagramSource)){
    showSource();
    return;
  }

  try{
    if(!window.mermaidInitialized){
      window.mermaid.initialize({
        startOnLoad:false,
        securityLevel:'strict',
        theme:document.body.classList.contains('dark')?'dark':'default',
        themeVariables:{fontSize:'12px'},
        flowchart:{nodeSpacing:10,rankSpacing:12,diagramPadding:4,useMaxWidth:true,htmlLabels:true,curve:'linear'}
      });
      window.mermaidInitialized=true;
    }
    const id=`grounded-diagram-${++groundedDiagramSequence}`;
    Promise.resolve(window.mermaid.render(id,diagramSource)).then(result=>{
      if(!container.isConnected)return;
      container.innerHTML=result.svg;
      if(result.bindFunctions)result.bindFunctions(container);
    }).catch(showSource);
  }catch(_){
    showSource();
  }
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
  box.innerHTML=`<textarea aria-label="Edit message"></textarea><div class="question-edit-actions"><button type="button" class="question-edit-cancel"><span class="ui-icon-slot">${window.leeLearnIcon('x')}</span><span>Cancel</span></button><button type="button" class="question-edit-send"><span class="ui-icon-slot">${window.leeLearnIcon('send')}</span><span>Send</span></button></div>`;
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

function normalizeEscapedMarkdown(text){
  return String(text).replace(/\r\n?/g,'\n')
    .replace(/^(\s*)\\(#{1,6})(?=\s)/gm,'$1$2')
    .replace(/^(\s*)\\([-+*])(?=\s)/gm,'$1$2')
    .replace(/^(\s*)\\(>)(?=\s)/gm,'$1$2')
    .replace(/^(\s*)\\([-*_]{3,})\s*$/gm,'$1$2')
    .replace(/^(\s*)\\(`{3,}|~{3,})(.*)$/gm,'$1$2$3')
    .replace(/\\\*\\\*(\S(?:.*?\S)?)\\\*\\\*/g,'**$1**')
    .replace(/\\\*\*(\S(?:.*?\S)?)\\\*\*/g,'**$1**')
    .replace(/\\\*([^*\s](?:[^*\n]*?[^*\s])?)\\\*/g,'*$1*')
    .replace(/\\`([^`\n]+)\\`/g,'`$1`');
}

function formatInlineMarkdown(text){
  let value=escapeHtml(text);
  const codeSpans=[];
  const mathSpans=[];
  value=value.replace(/`([^`\n]+)`/g,(_,code)=>{
    const token=`\u0000INLINECODE${codeSpans.length}\u0000`;
    codeSpans.push(`<code class="inline-code">${code}</code>`);
    return token;
  });
  value=value.replace(/\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\$(?!\$)[^$\n]+\$/g,math=>{
    const token=`\u0000INLINEMATH${mathSpans.length}\u0000`;
    mathSpans.push(math);
    return token;
  });
  const generatedMath=[];
  const wrapMath=(_,prefix,expression)=>{
    const token=`\u0000GENERATEDMATH${generatedMath.length}\u0000`;
    generatedMath.push(`$${expression}$`);
    return `${prefix}${token}`;
  };
  value=value.replace(/&lt;\s*br\s*\/?\s*&gt;|&lt;\s*\/br\s*&gt;/gi,'<br>');
  value=value.replace(/(^|[^\w$])(\(\s*[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+\s*,\s*[A-Za-z0-9]+\s*\))(?![\w$])/g,wrapMath);
  value=value.replace(/(^|[^\w$])((?:\d+\s*)?\(\s*[A-Za-z]\s*[-+]\s*\d+\s*\))(?![\w$])/g,wrapMath);
  value=value.replace(/(^|[^\w$])([A-Za-z]\s*[-+]\s*\d+)(?![\w$])/g,wrapMath);
  value=value.replace(/(^|[^\w$])([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+)(?![\w$])/g,(_,prefix,expression)=>`${prefix}$${expression}$`);
  value=value.replace(/\*\*(\S(?:.*?\S)?)\*\*/g,'<strong>$1</strong>');
  value=value.replace(/__((?:\S).*?\S)__/g,'<strong>$1</strong>');
  value=value.replace(/(^|[^\w*])\*([^*\n]+)\*(?!\*)/g,'$1<em>$2</em>');
  value=value.replace(/(^|[^\w_])_([^_\n]+)_(?!\w)/g,'$1<em>$2</em>');
  mathSpans.forEach((math,index)=>{value=value.replace(`\u0000INLINEMATH${index}\u0000`,math);});
  generatedMath.forEach((math,index)=>{value=value.replace(`\u0000GENERATEDMATH${index}\u0000`,math);});
  codeSpans.forEach((code,index)=>{value=value.replace(`\u0000INLINECODE${index}\u0000`,code);});
  return value;
}

function showReadableMath(element){
  const walker=document.createTreeWalker(element,NodeFilter.SHOW_TEXT);
  const textNodes=[];
  while(walker.nextNode()){
    if(!walker.currentNode.parentElement?.closest('pre,code,mjx-container'))textNodes.push(walker.currentNode);
  }
  textNodes.forEach(node=>{
    node.nodeValue=node.nodeValue.replace(/\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(?!\$)([^$\n]+)\$/g,(_,inline,display,block,dollar)=>inline??display??block??dollar);
  });
}

function typesetAnswerMath(element){
  const typeset=()=>{
    try{
      if(!window.MathJax?.typesetPromise){showReadableMath(element);return;}
      const pending=window.MathJax.typesetPromise([element]);
      if(pending&&typeof pending.catch==='function')pending.catch(()=>showReadableMath(element));
    }catch(_){showReadableMath(element);}
  };
  if(window.MathJax?.typesetPromise){typeset();return;}
  if(!window.aiLmsMathJaxLoadPromise){
    window.MathJax=window.MathJax||{
      tex:{inlineMath:[['$','$'],['\\(','\\)']],displayMath:[['$$','$$'],['\\[','\\]']],processEscapes:true},
      options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']},
      startup:{typeset:false}
    };
    window.aiLmsMathJaxLoadPromise=new Promise(resolve=>{
      const script=document.createElement('script');
      script.src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
      script.onload=resolve;
      script.onerror=resolve;
      document.head.appendChild(script);
    });
  }
  window.aiLmsMathJaxLoadPromise.then(typeset);
}

function splitMarkdownTableRow(line){
  return line.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(cell=>cell.trim());
}

function isMarkdownTableSeparator(line){
  const cells=splitMarkdownTableRow(line);
  return cells.length>1&&cells.every(cell=>/^:?-{3,}:?$/.test(cell));
}

function renderMarkdownTable(headerLine,bodyLines){
  const headers=splitMarkdownTableRow(headerLine);
  const rows=bodyLines.map(splitMarkdownTableRow);
  let html='<div class="table-scroll"><table class="answer-table"><thead><tr>';
  headers.forEach(cell=>html+=`<th>${formatInlineMarkdown(cell)}</th>`);
  html+='</tr></thead><tbody>';
  rows.forEach(row=>{
    html+='<tr>';
    headers.forEach((_,index)=>html+=`<td>${formatInlineMarkdown(row[index]||'')}</td>`);
    html+='</tr>';
  });
  return html+'</tbody></table></div>';
}

function renderMarkdownList(lines,start){
  const first=normalizeEscapedMarkdown(lines[start]).match(/^([ \t]*)([-+*]|\d+[.)])\s+(.+)$/);
  const baseIndent=first[1].replace(/\t/g,'    ').length;
  const ordered=/^\d/.test(first[2]);
  const tag=ordered?'ol':'ul';
  const items=[];
  let index=start;

  while(index<lines.length){
    const match=normalizeEscapedMarkdown(lines[index]).match(/^([ \t]*)([-+*]|\d+[.)])\s+(.+)$/);
    if(!match)break;
    const indent=match[1].replace(/\t/g,'    ').length;
    if(indent<baseIndent)break;
    if(indent>baseIndent){
      if(!items.length)break;
      const nested=renderMarkdownList(lines,index);
      if(nested.nextIndex===index)break;
      items[items.length-1].children+=nested.html;
      index=nested.nextIndex;
      continue;
    }
    if(/^\d/.test(match[2])!==ordered)break;
    items.push({text:match[3],children:''});
    index++;
  }

  const html=`<${tag}>${items.map(item=>`<li>${formatInlineMarkdown(item.text)}${item.children}</li>`).join('')}</${tag}>`;
  return {html,nextIndex:index};
}

function isAsciiBoxLine(line){
  return /^\s*\+[-=+]+\+\s*$/.test(line)||/^\s*\|[^\n]*\|\s*$/.test(line);
}

function isAsciiBoxDiagram(lines){
  return lines.filter(line=>/^\s*\+[-=+]+\+\s*$/.test(line)).length>=2
    &&lines.some(line=>/^\s*\|[^\n]*\|\s*$/.test(line));
}

function renderCodeBlock(code,language='',blockClass=''){
  const validLanguage=/^[a-zA-Z0-9_-]+$/.test(language);
  const safeLanguage=validLanguage?` language-${language}`:'';
  const safeBlockClass=blockClass==='fenced-code-block'?' fenced-code-block':'';
  const languageLabel=blockClass==='fenced-code-block'&&validLanguage?` data-language="${escapeHtml(language)}"`:'';
  return `<pre class="answer-code-block${safeBlockClass}"${languageLabel}><code class="${safeLanguage.trim()}">${escapeHtml(code)}</code></pre>`;
}

function formatAnswer(text){
  const sourceLines=String(text||'').replace(/\r\n?/g,'\n').split('\n');
  const fencedBlocks=[];
  const lines=[];
  for(let sourceIndex=0;sourceIndex<sourceLines.length;){
    const fence=normalizeEscapedMarkdown(sourceLines[sourceIndex]).match(/^\s*(`{3,}|~{3,})(.*)$/);
    if(!fence){
      lines.push(sourceLines[sourceIndex++]);
      continue;
    }
    const language=(fence[2].trim().match(/^[a-zA-Z0-9_-]+/)||[])[0]||'';
    const close=new RegExp(`^\\s*${fence[1][0]}{${fence[1].length},}\\s*$`);
    const code=[];
    sourceIndex++;
    while(sourceIndex<sourceLines.length&&!close.test(normalizeEscapedMarkdown(sourceLines[sourceIndex])))code.push(sourceLines[sourceIndex++]);
    if(sourceIndex<sourceLines.length)sourceIndex++;
    const token=`\u0000FENCEDCODE${fencedBlocks.length}\u0000`;
    fencedBlocks.push(renderCodeBlock(code.join('\n'),language,'fenced-code-block'));
    lines.push(token);
  }
  const output=[];
  let paragraph=[];
  let index=0;
  const flushParagraph=()=>{
    if(paragraph.length){
      output.push(`<p>${paragraph.map(formatInlineMarkdown).join('<br>')}</p>`);
      paragraph=[];
    }
  };

  while(index<lines.length){
    const line=normalizeEscapedMarkdown(lines[index]);
    const protectedCode=line.match(/^\u0000FENCEDCODE(\d+)\u0000$/);
    if(protectedCode){
      flushParagraph();
      output.push(fencedBlocks[Number(protectedCode[1])]);
      index++;
      continue;
    }

    const heading=line.match(/^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if(heading){
      flushParagraph();
      const level=heading[1].length;
      output.push(`<h${level}>${formatInlineMarkdown(heading[2])}</h${level}>`);
      index++;
      continue;
    }

    if(/^\s{0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(line)){
      flushParagraph();
      output.push('<hr>');
      index++;
      continue;
    }

    if(index+1<lines.length&&line.includes('|')&&isMarkdownTableSeparator(normalizeEscapedMarkdown(lines[index+1]))){
      flushParagraph();
      const body=[];
      index+=2;
      while(index<lines.length&&normalizeEscapedMarkdown(lines[index]).includes('|')&&lines[index].trim())body.push(normalizeEscapedMarkdown(lines[index++]));
      output.push(renderMarkdownTable(line,body));
      continue;
    }

    const quote=line.match(/^ {0,3}>\s?(.*)$/);
    if(quote){
      flushParagraph();
      const quoted=[];
      while(index<lines.length){
        const match=normalizeEscapedMarkdown(lines[index]).match(/^ {0,3}>\s?(.*)$/);
        if(!match)break;
        quoted.push(formatInlineMarkdown(match[1]));
        index++;
      }
      output.push(`<blockquote>${quoted.join('<br>')}</blockquote>`);
      continue;
    }

    if(/^\s*([-+*]|\d+[.)])\s+/.test(line)){
      flushParagraph();
      const list=renderMarkdownList(lines,index);
      output.push(list.html);
      index=list.nextIndex;
      continue;
    }

    if(isAsciiBoxLine(line)){
      let end=index;
      while(end<lines.length&&normalizeEscapedMarkdown(lines[end]).trim())end++;
      const diagram=lines.slice(index,end).map(normalizeEscapedMarkdown);
      if(isAsciiBoxDiagram(diagram)){
        flushParagraph();
        output.push(renderCodeBlock(diagram.join('\n'),'text'));
        index=end;
        continue;
      }
      paragraph.push(line);
      index++;
      continue;
    }

    if(!line.trim())flushParagraph();
    else paragraph.push(line);
    index++;
  }
  flushParagraph();
  return output.join('');
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
function applyTheme(){document.body.classList.toggle('dark',state.dark);$('themeBtn').innerHTML=window.leeLearnIcon(state.dark?'moon':'sun');}
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
async function downloadAnswerPdf(chatId,messageId){showToast('Preparing PDF...');const r=await fetch(`/api/chats/${chatId}/answer-pdf`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message_id:messageId})});if(!r.ok){const d=await r.json().catch(()=>({}));showToast(d.error||'Could not create PDF.');return;}const blob=await r.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='LeeLearn-Answer.pdf';a.click();URL.revokeObjectURL(url);showToast('PDF generated.');}
function showToast(text){const t=$('toast');t.textContent=text;t.classList.add('show');clearTimeout(showToast.timer);showToast.timer=setTimeout(()=>t.classList.remove('show'),2200);}

function showBackgroundCompletionNotification(chatId, answer){
  const chat=state.chats.find(c=>String(c.id)===String(chatId));
  const title=chat?.title || 'Lee AI Tutor';
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
      <button type="button" class="background-notification-close" aria-label="Dismiss notification">${window.leeLearnIcon('x')}</button>
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
    pop.innerHTML='<div class="youtube-popover-head"><span class="youtube-play-mini">'+actionIcon('youtube')+'</span><div><strong>YouTube recommendation</strong><small>No direct video was found</small></div><button class="youtube-close" type="button" aria-label="Close">'+actionIcon('x')+'</button></div><div class="youtube-empty">No suitable educational video was found for this topic.</div><button class="youtube-search" type="button">'+actionIcon('search')+'<span>Search this topic on YouTube</span></button>';
    pop.querySelector('.youtube-search').onclick=()=>{const active=state.chats.find(c=>String(c.id)===String(state.activeChatId));const msgs=active?.messages||[];let q='';for(let i=msgs.length-1;i>=0;i--){if(msgs[i].role==='user'){q=msgs[i].content||'';break;}}window.open('https://www.youtube.com/results?search_query='+encodeURIComponent(q),'_blank','noopener,noreferrer');};
  }else{
    const title=escapeHtml(video.title||'Recommended video'),channel=escapeHtml(video.channel||'YouTube'),url=String(video.url||'');
    pop.innerHTML=`<div class="youtube-popover-head"><span class="youtube-play-mini">${actionIcon('youtube')}</span><div><strong>Recommended video</strong><small>${channel}</small></div><button class="youtube-close" type="button" aria-label="Close">${actionIcon('x')}</button></div><div class="youtube-video-title">${title}</div><a class="youtube-watch" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Watch on YouTube</a>`;
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
  pop.innerHTML='<div class="sources-popover-head"><span class="sources-book">'+actionIcon('sources')+'</span><div><strong>Course Material Sources</strong><small>The material pages used for this answer</small></div><button class="sources-close" type="button" aria-label="Close">'+actionIcon('x')+'</button></div><div class="sources-list">'+unique.map(x=>`<div class="source-item"><strong>${escapeHtml(x.name)}</strong>${x.page?`<span>Page ${escapeHtml(x.page)}</span>`:''}${x.url?`<a href="${escapeHtml(x.url)}" target="_blank" rel="noopener noreferrer">Open source</a>`:''}</div>`).join('')+'</div>';
  pop.querySelector('.sources-close').onclick=closeSourcesPopover;
  pop.classList.add('open');pop.setAttribute('aria-hidden','false');
  const r=anchor.getBoundingClientRect(),width=Math.min(380,window.innerWidth-20);pop.style.width=width+'px';pop.style.left=Math.max(10,Math.min(window.innerWidth-width-10,r.left-width+40))+'px';pop.style.top='10px';const h=pop.offsetHeight;let top=r.bottom+8;if(top+h>window.innerHeight-10)top=r.top-h-8;if(top<10)top=10;pop.style.top=top+'px';
}
function closeSourcesPopover(){const p=$('sourcesPopover');if(p){p.classList.remove('open');p.setAttribute('aria-hidden','true');p.innerHTML='';}}
window.addEventListener('resize',()=>{closeYoutubePopover();closeSourcesPopover();},{passive:true});
window.addEventListener('click',e=>{const yp=$('youtubePopover'),sp=$('sourcesPopover');if(yp?.classList.contains('open')&&!yp.contains(e.target)&&!e.target.closest('[data-action="youtube"]'))closeYoutubePopover();if(sp?.classList.contains('open')&&!sp.contains(e.target)&&!e.target.closest('[data-action="sources"]'))closeSourcesPopover();});