import { useState, useRef, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';
import { 
  Send, Bot, User, LogOut, Menu, Plus, 
  Database, Users, X, Info, Sparkles 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
}

export default function Chat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    const fetchUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.user?.email) {
        setUserEmail(session.user.email);
      }
    };
    fetchUser();
  }, []);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    navigate('/login');
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const text = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setLoading(true);

    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      console.log('REQUEST_START', { question: text, top_k: 3 });
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(session && { Authorization: `Bearer ${session.access_token}` }),
        },
        body: JSON.stringify({ question: text, top_k: 3 }),
      });

      console.log('RESPONSE_STATUS', res.status, res.statusText);
      if (!res.ok) throw new Error(`Chat API failed: ${res.status} ${res.statusText}`);
      const data = await res.json();
      console.log('RESPONSE_JSON', data);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources?.map((s: string, i: number) => ({
          text: s,
          similarity: data.distances?.[i] || 0
        }))
      }]);
    } catch (err: any) {
      console.error('CHAT_ERROR', err);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Sorry, I encountered an error: ${err.message || 'Unknown error'}. Is the backend API running on port 8000?` 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const userInitial = userEmail ? userEmail.charAt(0).toUpperCase() : 'U';

  return (
    <div className="app-layout font-sans">
      {/* Sidebar Overlay for Mobile */}
      {isSidebarOpen && (
        <div 
          className="sidebar-overlay show block" 
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <Bot className="h-6 w-6" />
            <span>RAG Core Platform</span>
          </div>
          <button 
            onClick={() => setMessages([])}
            className="new-chat-btn"
          >
            <Plus className="h-4 w-4" />
            <span>New Session</span>
          </button>
        </div>

        <nav className="chat-list space-y-4 p-4">
          <div className="px-2 text-[10px] font-bold text-gray-500 uppercase tracking-widest">
            Workspace
          </div>
          <div className="space-y-1">
            <button 
              onClick={() => navigate('/kbs')} 
              className="w-full flex items-center px-3 py-2 text-sm rounded-lg text-gray-300 hover:bg-[#252a3a] hover:text-white transition-colors gap-3"
            >
              <Database className="h-4 w-4 text-gray-400" />
              <span>Knowledge Bases</span>
            </button>
            <button 
              onClick={() => navigate('/users')} 
              className="w-full flex items-center px-3 py-2 text-sm rounded-lg text-gray-300 hover:bg-[#252a3a] hover:text-white transition-colors gap-3"
            >
              <Users className="h-4 w-4 text-gray-400" />
              <span>User Manager</span>
            </button>
          </div>
        </nav>

        {/* Sidebar Footer */}
        <div className="sidebar-footer">
          <div className="sidebar-footer-avatar">{userInitial}</div>
          <div className="sidebar-footer-info">
            <div className="sidebar-footer-name truncate">{userEmail || 'Loading...'}</div>
            <div className="sidebar-footer-plan">Developer Tenant</div>
          </div>
          <button 
            onClick={handleLogout} 
            title="Log out"
            className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </aside>

      {/* Main Chat Workspace */}
      <main className="main-area">
        {/* Top Header */}
        <header className="top-header">
          <div className="top-header-left">
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="mobile-menu-btn"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="top-header-title flex items-center space-x-2">
              <span>Interactive RAG Session</span>
            </div>
          </div>
          
          <div className="top-header-right">
            <div className="model-badge">
              <div className="model-badge-dot" />
              <span>Gemini 2.0 Flash</span>
            </div>
          </div>
        </header>

        {/* Chat Messages */}
        <div className="messages-area">
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="inline-flex p-4 bg-[#7c3aed]/10 rounded-2xl mb-6">
                  <Sparkles className="h-10 w-10 text-[#7c3aed]" />
                </div>
                <h2>Ask RAG Assistant</h2>
                <p>
                  Query document indices, retrieve contextual matches, and synthesize generative insights.
                </p>
                <div className="empty-state-hints">
                  <button 
                    onClick={() => {
                      setInput('What is RAG?');
                    }}
                    className="hint-chip"
                  >
                    What is RAG?
                  </button>
                  <button 
                    onClick={() => {
                      setInput('Summarize my uploaded document content');
                    }}
                    className="hint-chip"
                  >
                    Summarize my files
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <AnimatePresence initial={false}>
                  {messages.map((msg, i) => (
                    <motion.div 
                      key={i}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                      className={`message-row ${msg.role === 'user' ? 'user' : 'assistant'}`}
                    >
                      <div className="message-content">
                        <div className={`message-bubble ${msg.role === 'user' ? 'user-bubble' : 'assistant-bubble'}`}>
                          <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                          
                          {/* Display Retrieval Sources if present */}
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="sources-container border-t border-[#282d3e] pt-3 mt-3 space-y-2">
                              <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest flex items-center">
                                <Info className="h-3 w-3 mr-1.5 text-[#8b5cf6]" />
                                Context Retrieval Matches
                              </div>
                              <div className="space-y-1.5">
                                {msg.sources.map((src, idx) => (
                                  <details key={idx} className="source-card">
                                    <summary>
                                      <span className="source-label">Source Segment {idx + 1}</span>
                                      <span className="score-badge">
                                        Match Score: {(src.similarity * 100).toFixed(0)}%
                                      </span>
                                    </summary>
                                    <div className="source-text">{src.text}</div>
                                  </details>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            )}
            
            {/* Animated Typing Indicator */}
            {loading && (
              <div className="message-row assistant">
                <div className="message-content">
                  <div className="message-bubble assistant-bubble">
                    <div className="typing-indicator">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Bar Area */}
        <div className="input-area">
          <div className="input-wrapper">
            <div className="input-top">
              <textarea
                rows={1}
                placeholder="Ask your documents anything..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="send-btn"
                title="Send query"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
          <div className="input-hint">
            Press <kbd>Enter</kbd> to submit, <kbd>Shift + Enter</kbd> for new line.
          </div>
        </div>
      </main>
    </div>
  );
}
