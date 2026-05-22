// src/main.ts — Production-grade AI Chat UI with multi-session support
import './style.css';

// ───────────────────────── SVG Icons ─────────────────────────
const ICONS = {
  logo: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`,
  plus: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  chat: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
  trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
  send: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`,
  sparkle: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>`,
  source: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`,
  chevron: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`,
  menu: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>`,
  user: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
  bot: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>`,
};

// ───────────────────────── Types ─────────────────────────
interface Source {
  text: string;
  similarity: number;
  source: string;
}
interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}
interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  messages: Message[];
}

// ───────────────────────── State ─────────────────────────
const STORAGE_KEY = 'rag_chat_sessions';
let sessions: ChatSession[] = [];
let activeChatId: string | null = null;
let searchQuery = '';

// ───────────────────────── Utilities ─────────────────────────
function uuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function saveSessions(): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function loadSessions(): void {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw) {
    try { sessions = JSON.parse(raw) as ChatSession[]; } catch { sessions = []; }
  }
  if (!sessions.length) {
    const s = createSession();
    sessions.push(s);
    activeChatId = s.id;
    saveSessions();
  } else if (!activeChatId) {
    activeChatId = sessions[0].id;
  }
}

function createSession(): ChatSession {
  return { id: uuid(), title: 'New Chat', createdAt: Date.now(), messages: [] };
}

function getActiveSession(): ChatSession {
  const c = sessions.find(s => s.id === activeChatId);
  if (!c) { activeChatId = sessions[0].id; return sessions[0]; }
  return c;
}

function generateTitle(msg: string): string {
  const t = msg.trim();
  if (!t) return 'New Chat';
  const w = t.split(' ').slice(0, 6).join(' ');
  return w.length > 35 ? w.slice(0, 32) + '…' : w;
}

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

// ───────────────────────── DOM Construction ─────────────────────────
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
<div class="app-layout">
  <!-- Sidebar -->
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo">
        ${ICONS.logo}
        <span>RAG Assistant</span>
      </div>
      <button class="new-chat-btn" id="newChatBtn">
        ${ICONS.plus}
        <span>New Chat</span>
      </button>
      <div class="search-box">
        ${ICONS.search}
        <input type="text" id="searchInput" placeholder="Search chats…" />
      </div>
    </div>
    <div class="chat-list" id="chatList"></div>
    <div class="sidebar-footer">
      <div class="sidebar-footer-avatar">U</div>
      <div class="sidebar-footer-info">
        <div class="sidebar-footer-name">Researcher</div>
        <div class="sidebar-footer-plan">RAG Pro</div>
      </div>
    </div>
  </aside>

  <!-- Sidebar overlay for mobile -->
  <div class="sidebar-overlay" id="sidebarOverlay"></div>

  <!-- Main Area -->
  <div class="main-area">
    <header class="top-header">
      <div class="top-header-left">
        <button class="mobile-menu-btn" id="mobileMenuBtn">${ICONS.menu}</button>
        <span class="top-header-title" id="headerTitle">New Chat</span>
      </div>
      <div class="top-header-right">
        <div class="model-badge">
          <span class="model-badge-dot"></span>
          RAG Assistant
        </div>
      </div>
    </header>

    <div class="messages-area" id="messagesArea">
      <div class="messages-container" id="messagesContainer"></div>
    </div>

    <div class="input-area">
      <div class="input-wrapper">
        <div class="input-top">
          <textarea id="userInput" rows="1" placeholder="Ask anything about your research…"></textarea>
          <button class="send-btn" id="sendBtn" disabled>${ICONS.send}</button>
        </div>
      </div>
      <div class="input-hint">
        <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for new line
      </div>
    </div>
  </div>
</div>`;

// ───────────────────────── Element Refs ─────────────────────────
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const sidebar       = $<HTMLElement>('sidebar');
const sidebarOverlay= $<HTMLElement>('sidebarOverlay');
const mobileMenuBtn = $<HTMLButtonElement>('mobileMenuBtn');
const chatListEl    = $<HTMLElement>('chatList');
const searchInput   = $<HTMLInputElement>('searchInput');
const newChatBtn    = $<HTMLButtonElement>('newChatBtn');
const headerTitle   = $<HTMLElement>('headerTitle');
const messagesArea  = $<HTMLElement>('messagesArea');
const msgContainer  = $<HTMLElement>('messagesContainer');
const userInput     = $<HTMLTextAreaElement>('userInput');
const sendBtn       = $<HTMLButtonElement>('sendBtn');

// ───────────────────────── Rendering: Chat List ─────────────────────────
function renderChatList(): void {
  chatListEl.innerHTML = '';
  const filtered = searchQuery
    ? sessions.filter(s => s.title.toLowerCase().includes(searchQuery.toLowerCase()))
    : sessions;

  filtered.forEach(chat => {
    const isActive = chat.id === activeChatId;
    const item = document.createElement('div');
    item.className = `chat-item${isActive ? ' active' : ''}`;
    item.innerHTML = `
      <span class="chat-item-icon">${ICONS.chat}</span>
      <div class="chat-item-content">
        <div class="chat-item-title">${escapeHtml(chat.title)}</div>
        <div class="chat-item-time">${relativeTime(chat.createdAt)}</div>
      </div>`;

    const delBtn = document.createElement('button');
    delBtn.className = 'chat-item-delete';
    delBtn.innerHTML = ICONS.trash;
    delBtn.title = 'Delete chat';
    delBtn.addEventListener('click', e => {
      e.stopPropagation();
      sessions = sessions.filter(c => c.id !== chat.id);
      if (!sessions.length) {
        const s = createSession();
        sessions.push(s);
        activeChatId = s.id;
      } else if (activeChatId === chat.id) {
        activeChatId = sessions[0].id;
      }
      saveSessions();
      renderChatList();
      renderMessages();
      updateHeader();
    });

    item.appendChild(delBtn);
    item.addEventListener('click', () => {
      activeChatId = chat.id;
      renderChatList();
      renderMessages();
      updateHeader();
      closeMobileSidebar();
    });
    chatListEl.appendChild(item);
  });
}

// ───────────────────────── Rendering: Messages ─────────────────────────
function renderMessages(): void {
  msgContainer.innerHTML = '';
  const chat = getActiveSession();

  if (!chat.messages.length) {
    renderEmptyState();
    return;
  }

  chat.messages.forEach(msg => msgContainer.appendChild(createMessageEl(msg)));
  scrollToBottom();
}

function renderEmptyState(): void {
  msgContainer.innerHTML = `
    <div class="empty-state">
      <div class="empty-state-icon">${ICONS.sparkle}</div>
      <h2>What can I help with?</h2>
      <p>Ask questions about your research documents. I'll retrieve the most relevant information from your knowledge base.</p>
      <div class="empty-state-hints">
        <div class="hint-chip" data-hint="Summarize the latest findings on CRISPR gene editing">Summarize CRISPR findings</div>
        <div class="hint-chip" data-hint="What are the key differences between transformers and RNNs?">Transformers vs RNNs</div>
        <div class="hint-chip" data-hint="Explain the mechanism of mRNA vaccines">mRNA vaccine mechanism</div>
      </div>
    </div>`;

  // Bind hint chips
  msgContainer.querySelectorAll('.hint-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const hint = (chip as HTMLElement).dataset.hint || '';
      userInput.value = hint;
      userInput.dispatchEvent(new Event('input'));
      userInput.focus();
    });
  });
}

function createMessageEl(msg: Message): HTMLElement {
  const row = document.createElement('div');
  row.className = `message-row ${msg.role}`;

  if (msg.role === 'assistant') {
    row.innerHTML = `<div class="message-avatar assistant-av">${ICONS.bot}</div>`;
  }

  const content = document.createElement('div');
  content.className = 'message-content';

  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${msg.role === 'user' ? 'user-bubble' : 'assistant-bubble'}`;
  bubble.innerHTML = formatContent(msg.content);
  content.appendChild(bubble);

  // Source cards
  if (msg.role === 'assistant' && msg.sources?.length) {
    const srcWrap = document.createElement('div');
    srcWrap.className = 'sources-container';
    msg.sources.forEach((src, idx) => {
      const card = document.createElement('details');
      card.className = 'source-card';
      card.innerHTML = `
        <summary>
          <span class="source-label">${ICONS.source} Source ${idx + 1}</span>
          <span style="display:flex;align-items:center;gap:6px;">
            <span class="score-badge">${(src.similarity * 100).toFixed(0)}% match</span>
            <span class="chevron-icon">${ICONS.chevron}</span>
          </span>
        </summary>
        <div class="source-text">${escapeHtml(src.text)}</div>`;
      srcWrap.appendChild(card);
    });
    content.appendChild(srcWrap);
  }

  row.appendChild(content);

  if (msg.role === 'user') {
    const av = document.createElement('div');
    av.className = 'message-avatar user-av';
    av.innerHTML = ICONS.user;
    row.appendChild(av);
  }

  return row;
}

function formatContent(text: string): string {
  return escapeHtml(text).replace(/\n/g, '<br/>');
}

function escapeHtml(s: string): string {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function scrollToBottom(): void {
  requestAnimationFrame(() => {
    messagesArea.scrollTop = messagesArea.scrollHeight;
  });
}

function updateHeader(): void {
  const chat = getActiveSession();
  headerTitle.textContent = chat.title;
}

// ───────────────────────── Typing Indicator ─────────────────────────
function showTypingIndicator(): HTMLElement {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.id = 'typingRow';
  row.innerHTML = `
    <div class="message-avatar assistant-av">${ICONS.bot}</div>
    <div class="message-content">
      <div class="message-bubble assistant-bubble">
        <div class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    </div>`;
  msgContainer.appendChild(row);
  scrollToBottom();
  return row;
}

// ───────────────────────── Mobile Sidebar ─────────────────────────
function toggleMobileSidebar(): void {
  sidebar.classList.toggle('open');
}
function closeMobileSidebar(): void {
  sidebar.classList.remove('open');
}

// ───────────────────────── Auto-resize Textarea ─────────────────────────
function autoResize(): void {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + 'px';
}

// ───────────────────────── Event Bindings ─────────────────────────
newChatBtn.addEventListener('click', () => {
  const s = createSession();
  sessions.unshift(s);
  activeChatId = s.id;
  saveSessions();
  renderChatList();
  renderMessages();
  updateHeader();
  userInput.focus();
  closeMobileSidebar();
});

searchInput.addEventListener('input', () => {
  searchQuery = searchInput.value;
  renderChatList();
});

mobileMenuBtn.addEventListener('click', toggleMobileSidebar);
sidebarOverlay.addEventListener('click', closeMobileSidebar);

userInput.addEventListener('input', () => {
  sendBtn.disabled = !userInput.value.trim();
  autoResize();
});

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', () => sendMessage());

// ───────────────────────── Send Message ─────────────────────────
async function sendMessage(): Promise<void> {
  const text = userInput.value.trim();
  if (!text) return;

  const chat = getActiveSession();
  chat.messages.push({ role: 'user', content: text });

  if (chat.title === 'New Chat') {
    chat.title = generateTitle(text);
  }

  saveSessions();
  renderMessages();
  renderChatList();
  updateHeader();

  // Reset input
  userInput.value = '';
  userInput.style.height = 'auto';
  sendBtn.disabled = true;

  // Show typing
  const typingEl = showTypingIndicator();

  try {
    const resp = await fetch('http://127.0.0.1:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, top_k: 3 }),
    });
    const data = await resp.json();

    const assistantMsg: Message = {
      role: 'assistant',
      content: data.answer,
      sources: data.sources?.map((s: string, i: number) => ({
        text: s,
        similarity: data.distances?.[i] ?? 0,
        source: 'Source',
      })),
    };

    typingEl.remove();
    chat.messages.push(assistantMsg);
    saveSessions();
    renderMessages();
    renderChatList();
    updateHeader();
  } catch (err) {
    console.error('Chat error:', err);
    typingEl.remove();

    // Show inline error
    const errRow = document.createElement('div');
    errRow.className = 'message-row assistant';
    errRow.innerHTML = `
      <div class="message-avatar assistant-av">${ICONS.bot}</div>
      <div class="message-content">
        <div class="message-bubble assistant-bubble" style="border-color:rgba(239,68,68,0.3);color:#fca5a5;">
          Failed to reach the server. Please check that the backend is running on port 8000.
        </div>
      </div>`;
    msgContainer.appendChild(errRow);
    scrollToBottom();
  } finally {
    sendBtn.disabled = !userInput.value.trim();
  }
}

// ───────────────────────── Init ─────────────────────────
loadSessions();
renderChatList();
renderMessages();
updateHeader();
userInput.focus();
