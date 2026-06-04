import { useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';
import { 
  LogOut, Database, Plus, Sparkles, Folder, 
  Calendar, FileText, Globe, AlertCircle, ArrowLeft, RefreshCw
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function KnowledgeBases() {
  const navigate = useNavigate();
  const [kbs, setKbs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  
  // Creation modal/form state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newKbName, setNewKbName] = useState('');
  const [newKbDesc, setNewKbDesc] = useState('');
  const [createError, setCreateError] = useState('');
  const [creating, setCreating] = useState(false);

  const fetchKbs = async () => {
    setLoading(true);
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;
    try {
      const res = await fetch('/api/kb', {
        headers: {
          Authorization: `Bearer ${session.access_token}`,
        }
      });
      if (res.ok) {
        const data = await res.json();
        setKbs(data);
      }
    } catch (err) {
      console.error('Failed to fetch KBs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKbs();
    
    // Fetch current user email
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

  const handleCreateKb = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKbName.trim()) {
      setCreateError('Name is required.');
      return;
    }
    
    setCreating(true);
    setCreateError('');
    
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error('Not authenticated');

      const res = await fetch('/api/kb', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          name: newKbName.trim(),
          description: newKbDesc.trim(),
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to create knowledge base');
      }

      // Refresh KB list
      await fetchKbs();
      
      // Close modal & reset fields
      setShowCreateModal(false);
      setNewKbName('');
      setNewKbDesc('');
    } catch (err: any) {
      setCreateError(err.message || 'Error occurred');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f1117] text-white flex flex-col font-sans relative overflow-hidden">
      {/* Dynamic Background Glows */}
      <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-[#7c3aed]/5 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-500/5 blur-[120px] pointer-events-none" />

      {/* Header */}
      <header className="bg-[#171923]/80 backdrop-blur-xl border-b border-[#2a2f3d] px-6 py-4 flex justify-between items-center sticky top-0 z-40">
        <div className="flex items-center space-x-6">
          <div onClick={() => navigate('/')} className="flex items-center space-x-2 cursor-pointer group">
            <ArrowLeft className="h-4 w-4 text-gray-400 group-hover:text-white transition-colors" />
            <h1 className="text-base font-bold text-white tracking-tight group-hover:text-purple-400 transition-colors">
              RAG SaaS Platform
            </h1>
          </div>
          <nav className="hidden md:flex space-x-1">
            <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-colors">Chat</button>
            <button className="bg-[#252a3a] text-white px-3 py-1.5 rounded-lg text-sm font-medium border border-[#2a2f3d]">Knowledge Bases</button>
            <button onClick={() => navigate('/users')} className="text-gray-400 hover:text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-colors">Users</button>
          </nav>
        </div>
        <div className="flex items-center space-x-4">
          <span className="text-xs text-gray-400 hidden sm:inline-block max-w-[150px] truncate">
            {userEmail}
          </span>
          <button 
            onClick={handleLogout} 
            className="flex items-center text-xs font-semibold text-gray-400 hover:text-red-400 bg-[#1e2230] border border-[#2a2f3d] hover:border-red-500/30 px-3 py-1.5 rounded-lg transition-all duration-200"
          >
            <LogOut className="h-3.5 w-3.5 mr-1.5" />
            Logout
          </button>
        </div>
      </header>

      {/* Main Workspace Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 space-y-8 z-10">
        {/* Banner Section */}
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center tracking-tight">
              <Database className="h-6 w-6 mr-3 text-[#8b5cf6]" />
              Knowledge Bases
            </h2>
            <p className="text-xs text-gray-400 mt-1">Manage document pipelines, query embeddings, and sync database catalogs.</p>
          </div>
          <div className="flex items-center space-x-2">
            <button 
              onClick={fetchKbs}
              title="Refresh list"
              className="p-2 bg-[#1e2230] border border-[#2a2f3d] hover:border-gray-500 rounded-lg text-gray-400 hover:text-white transition-all"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button 
              onClick={() => setShowCreateModal(true)}
              className="flex items-center justify-center bg-gradient-to-r from-[#7c3aed] to-blue-600 hover:from-[#8b5cf6] hover:to-blue-500 text-white px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-200 shadow-md shadow-[#7c3aed]/10"
            >
              <Plus className="h-4 w-4 mr-1.5" />
              New Catalog
            </button>
          </div>
        </div>

        {/* Dashboard KPI Stat Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-[#1e2230]/50 backdrop-blur border border-[#2a2f3d] rounded-xl p-5 flex items-center space-x-4">
            <div className="h-10 w-10 bg-purple-500/10 rounded-lg flex items-center justify-center text-[#8b5cf6]">
              <Folder className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">{kbs.length}</div>
              <div className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">Total Catalogs</div>
            </div>
          </div>
          <div className="bg-[#1e2230]/50 backdrop-blur border border-[#2a2f3d] rounded-xl p-5 flex items-center space-x-4">
            <div className="h-10 w-10 bg-blue-500/10 rounded-lg flex items-center justify-center text-blue-400">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">
                {kbs.reduce((acc, curr) => acc + (curr.document_count || 0), 0)}
              </div>
              <div className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">Indexed Files</div>
            </div>
          </div>
          <div className="bg-[#1e2230]/50 backdrop-blur border border-[#2a2f3d] rounded-xl p-5 flex items-center space-x-4">
            <div className="h-10 w-10 bg-green-500/10 rounded-lg flex items-center justify-center text-green-400">
              <Globe className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">Active</div>
              <div className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">Cluster Status</div>
            </div>
          </div>
        </div>

        {/* Catalog lists */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-[#1e2230]/30 border border-[#2a2f3d] rounded-xl p-6 space-y-4 animate-pulse">
                <div className="h-5 bg-gray-800 rounded w-2/3" />
                <div className="h-4 bg-gray-800 rounded w-full" />
                <div className="h-4 bg-gray-800 rounded w-5/6" />
                <div className="pt-2 flex justify-between">
                  <div className="h-3 bg-gray-800 rounded w-1/4" />
                  <div className="h-3 bg-gray-800 rounded w-1/3" />
                </div>
              </div>
            ))}
          </div>
        ) : kbs.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-16 bg-[#1e2230]/30 backdrop-blur border border-[#2a2f3d] rounded-2xl max-w-2xl mx-auto"
          >
            <Database className="h-12 w-12 text-gray-500 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-bold text-white tracking-tight">No Knowledge Bases Found</h3>
            <p className="text-xs text-gray-400 max-w-sm mx-auto mt-2 leading-relaxed">
              Create your first knowledge base catalog, sync your corporate documents, and start querying embeddings with Gemini instantly.
            </p>
            <button 
              onClick={() => setShowCreateModal(true)}
              className="mt-6 inline-flex items-center bg-[#7c3aed] hover:bg-[#8b5cf6] text-white text-xs font-semibold px-4 py-2 rounded-lg transition-all"
            >
              <Plus className="h-3.5 w-3.5 mr-1" />
              Create First Catalog
            </button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {kbs.map((kb, i) => (
              <motion.div 
                key={i} 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="bg-[#1e2230]/60 backdrop-blur border border-[#2a2f3d] hover:border-gray-500/50 rounded-xl p-6 flex flex-col justify-between transition-all duration-200 cursor-pointer shadow-lg hover:shadow-xl relative overflow-hidden group"
              >
                {/* Visual hover border decoration */}
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-purple-500 to-blue-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                
                <div>
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="text-base font-bold text-white tracking-tight group-hover:text-purple-400 transition-colors">
                      {kb.name}
                    </h3>
                    <span className="px-2 py-0.5 bg-purple-500/10 border border-purple-500/20 text-[#a78bfa] rounded-full text-[10px] font-semibold">
                      {kb.status || 'SYNCED'}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 line-clamp-3 mb-6 leading-relaxed">
                    {kb.description || 'No description cataloged for this knowledge repository.'}
                  </p>
                </div>
                
                <div className="flex justify-between items-center text-[10px] text-gray-500 border-t border-[#2a2f3d] pt-3 mt-auto">
                  <span className="flex items-center">
                    <Calendar className="h-3.5 w-3.5 mr-1 text-gray-600" />
                    {new Date(kb.created_at || Date.now()).toLocaleDateString()}
                  </span>
                  <span className="flex items-center">
                    <FileText className="h-3.5 w-3.5 mr-1 text-gray-600" />
                    {kb.document_count || 0} items
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </main>

      {/* Creation Modal Overlay */}
      <AnimatePresence>
        {showCreateModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Modal backdrop overlay */}
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => !creating && setShowCreateModal(false)}
            />
            
            {/* Modal Content */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 15 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 15 }}
              className="bg-[#1e2230] border border-[#2a2f3d] rounded-2xl w-full max-w-md p-6 shadow-2xl relative z-10"
            >
              <div className="flex items-center space-x-3 mb-5">
                <div className="h-10 w-10 rounded-xl bg-purple-500/10 flex items-center justify-center text-[#8b5cf6]">
                  <Database className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white tracking-tight">Create Knowledge Base</h3>
                  <p className="text-xs text-gray-400">Initialize a new isolated document catalog</p>
                </div>
              </div>

              <form onSubmit={handleCreateKb} className="space-y-4">
                {createError && (
                  <div className="bg-red-500/10 border border-red-500/30 text-red-400 p-3 rounded-lg text-xs flex items-start space-x-2">
                    <AlertCircle className="h-4 w-4 shrink-0 text-red-500" />
                    <span>{createError}</span>
                  </div>
                )}

                <div className="space-y-1">
                  <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                    Catalog Name
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Q3 Financial Research"
                    disabled={creating}
                    className="w-full px-3 py-2 bg-[#1a1f2e] border border-[#2a2f3d] rounded-lg text-sm text-white placeholder-gray-500 outline-none transition-all focus:border-[#7c3aed] focus:ring-1 focus:ring-[#7c3aed]"
                    value={newKbName}
                    onChange={(e) => setNewKbName(e.target.value)}
                  />
                </div>

                <div className="space-y-1">
                  <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                    Description
                  </label>
                  <textarea
                    rows={3}
                    placeholder="Provide a summary description of the files that will be cataloged..."
                    disabled={creating}
                    className="w-full px-3 py-2 bg-[#1a1f2e] border border-[#2a2f3d] rounded-lg text-sm text-white placeholder-gray-500 outline-none resize-none transition-all focus:border-[#7c3aed] focus:ring-1 focus:ring-[#7c3aed]"
                    value={newKbDesc}
                    onChange={(e) => setNewKbDesc(e.target.value)}
                  />
                </div>

                <div className="flex justify-end space-x-3 pt-3">
                  <button
                    type="button"
                    disabled={creating}
                    onClick={() => setShowCreateModal(false)}
                    className="px-4 py-2 bg-[#171923] hover:bg-[#252a3a] border border-[#2a2f3d] text-gray-300 rounded-lg text-xs font-semibold transition-all"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={creating}
                    className="flex items-center bg-gradient-to-r from-[#7c3aed] to-blue-600 hover:from-[#8b5cf6] hover:to-blue-500 text-white px-4 py-2 rounded-lg text-xs font-semibold transition-all shadow-md shadow-[#7c3aed]/10"
                  >
                    {creating ? (
                      <div className="flex items-center space-x-2">
                        <div className="h-3.5 w-3.5 rounded-full border-2 border-white/35 border-t-white animate-spin" />
                        <span>Creating...</span>
                      </div>
                    ) : (
                      'Create catalog'
                    )}
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
