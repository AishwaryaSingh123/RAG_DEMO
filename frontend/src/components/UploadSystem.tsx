import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { 
  FileText, Upload, Trash2, Globe, MessageSquare, 
  Database, RefreshCw, AlertCircle, FileUp
} from 'lucide-react';
import { renderSafe } from '../lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface Document {
  id: string;
  filename: string;
  source_type: string;
  kb_id: string | null;
  upload_date: string | null;
}

interface UploadSystemProps {
  currentChatId?: string | null;
  currentProjectId?: string | null;
  onUploadSuccess?: () => void;
}

export default function UploadSystem({ 
  currentChatId = null, 
  currentProjectId = null,
  onUploadSuccess
}: UploadSystemProps) {
  const [activeTab, setActiveTab] = useState<'global' | 'chat' | 'project'>('global');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Fetch all documents from the backend
  const fetchDocuments = async () => {
    setLoading(true);
    setError('');
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch('/api/documents', {
        headers: {
          Authorization: `Bearer ${session.access_token}`
        }
      });
      if (res.ok) {
        const data = await res.json();
        console.log("DEBUG: documents list fetched:", data);
        setDocuments(data);
      } else {
        throw new Error('Failed to load documents');
      }
    } catch (err: any) {
      console.error(err);
      setError('Could not sync files library from server.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Parse document scope from source_type
  const getDocumentScope = (doc: Document) => {
    const type = doc.source_type || 'File';
    if (type.startsWith('chat_')) {
      return { scope: 'chat', id: type.replace('chat_', '') };
    }
    if (type.startsWith('project_')) {
      return { scope: 'project', id: type.replace('project_', '') };
    }
    return { scope: 'global', id: null };
  };

  // Filter documents based on active tab and scoped IDs
  const filteredDocs = documents.filter(doc => {
    const info = getDocumentScope(doc);
    if (activeTab === 'global') {
      return info.scope === 'global';
    }
    if (activeTab === 'chat') {
      // If we have a currentChatId, filter by it; otherwise show all chat-specific files
      return info.scope === 'chat' && (!currentChatId || info.id === currentChatId);
    }
    if (activeTab === 'project') {
      // If we have a currentProjectId, filter by it; otherwise show all project-specific files
      return info.scope === 'project' && (!currentProjectId || info.id === currentProjectId);
    }
    return false;
  });

  // Handle file upload
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    setUploading(true);
    setProgress(15);
    setError('');
    setSuccess('');

    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error('You must be signed in.');

      const formData = new FormData();
      formData.append('file', file);
      
      // Map scope to backend source_type parameter
      let scopeType = 'File';
      if (activeTab === 'chat' && currentChatId) {
        scopeType = `chat_${currentChatId}`;
      } else if (activeTab === 'project' && currentProjectId) {
        scopeType = `project_${currentProjectId}`;
      } else if (activeTab === 'global') {
        scopeType = 'global';
      }
      formData.append('source_type', scopeType);

      setProgress(45);

      const res = await fetch('/api/rag/ingest', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${session.access_token}`
        },
        body: formData
      });

      setProgress(85);

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Ingestion failed');
      }

      setSuccess(`File "${file.name}" ingested successfully!`);
      setProgress(100);
      
      // Refresh documents
      await fetchDocuments();
      if (onUploadSuccess) onUploadSuccess();
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Error occurred during ingestion.');
    } finally {
      setTimeout(() => {
        setUploading(false);
        setProgress(0);
      }, 1000);
    }
  };

  // Handle file deletion
  const handleDelete = async (filename: string) => {
    if (!confirm(`Are you sure you want to remove "${filename}" from the RAG library?`)) return;

    setError('');
    setSuccess('');
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${session.access_token}`
        }
      });

      if (res.ok) {
        setSuccess(`Removed "${filename}" successfully.`);
        await fetchDocuments();
        if (onUploadSuccess) onUploadSuccess();
      } else {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to delete');
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Error occurred during deletion.');
    }
  };

  return (
    <div className="bg-[#1e2230]/50 backdrop-blur-md border border-[#2a2f3d] rounded-2xl p-6 shadow-xl space-y-6">
      {/* Selector Tabs */}
      <div className="flex justify-between items-center border-b border-[#2a2f3d] pb-4 flex-wrap gap-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center">
            <FileUp className="h-5 w-5 mr-2 text-[#8b5cf6]" />
            Document Ingestion Library
          </h3>
          <p className="text-xs text-gray-400 mt-1">Ingest text, docs, and pdf files into the vectorized RAG index.</p>
        </div>
        <div className="flex bg-[#171923] p-1 rounded-xl border border-[#2a2f3d]">
          <button
            onClick={() => setActiveTab('global')}
            className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === 'global' 
                ? 'bg-[#7c3aed] text-white' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Globe className="h-3.5 w-3.5" />
            <span>Global</span>
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === 'chat' 
                ? 'bg-[#7c3aed] text-white' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5" />
            <span>Chat</span>
          </button>
          <button
            onClick={() => setActiveTab('project')}
            className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === 'project' 
                ? 'bg-[#7c3aed] text-white' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Database className="h-3.5 w-3.5" />
            <span>Project</span>
          </button>
        </div>
      </div>

      {/* Alert panels */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 p-3 rounded-lg text-xs flex items-start space-x-2">
          <AlertCircle className="h-4 w-4 text-red-500 shrink-0" />
          <span>{renderSafe(error)}</span>
        </div>
      )}
      {success && (
        <div className="bg-green-500/10 border border-green-500/30 text-green-400 p-3 rounded-lg text-xs flex items-start space-x-2 animate-pulse">
          <div className="h-4 w-4 rounded-full bg-green-500/20 text-green-500 flex items-center justify-center shrink-0">✓</div>
          <span>{renderSafe(success)}</span>
        </div>
      )}

      {/* Upload zone */}
      <div className="border-2 border-dashed border-[#2a2f3d] hover:border-purple-500/40 rounded-xl p-8 text-center transition-all relative overflow-hidden group">
        <input
          type="file"
          id="scope-file-input"
          accept=".txt,.pdf,.docx"
          className="absolute inset-0 opacity-0 cursor-pointer"
          onChange={handleUpload}
          disabled={uploading}
        />
        
        {uploading ? (
          <div className="space-y-3">
            <RefreshCw className="h-10 w-10 text-[#8b5cf6] mx-auto animate-spin" />
            <h4 className="text-sm font-bold text-white">Ingesting File...</h4>
            <div className="max-w-xs mx-auto bg-gray-800 rounded-full h-1.5 overflow-hidden">
              <div 
                className="bg-gradient-to-r from-purple-500 to-blue-500 h-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-[10px] text-gray-500 font-semibold">{progress}% completed</p>
          </div>
        ) : (
          <div className="space-y-2">
            <Upload className="h-10 w-10 text-gray-500 mx-auto group-hover:text-[#8b5cf6] transition-colors" />
            <h4 className="text-sm font-bold text-white">
              Drag & drop file or <span className="text-[#8b5cf6]">browse</span>
            </h4>
            <p className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">
              TXT, PDF, or DOCX formats accepted (Max 10MB)
            </p>
            <p className="text-[10px] text-purple-400 mt-2 font-medium">
              Uploading to: <span className="uppercase font-bold">{activeTab} scope</span>
              {activeTab === 'chat' && currentChatId && ` (${currentChatId.slice(0, 8)})`}
              {activeTab === 'project' && currentProjectId && ` (${currentProjectId.slice(0, 8)})`}
            </p>
          </div>
        )}
      </div>

      {/* Files list */}
      <div className="space-y-3">
        <div className="flex justify-between items-center text-xs text-gray-500 font-semibold uppercase tracking-wider">
          <span>Files list</span>
          <button 
            onClick={fetchDocuments}
            title="Refresh files library"
            className="p-1 hover:text-white rounded transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>

        {loading ? (
          <div className="space-y-2">
            {[1, 2].map((i) => (
              <div key={i} className="h-12 bg-[#1e2230]/40 rounded-lg animate-pulse" />
            ))}
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="text-center py-8 bg-[#1e2230]/20 rounded-xl border border-[#2a2f3d]/50 text-gray-500 text-xs">
            <FileText className="h-8 w-8 mx-auto mb-2 opacity-30" />
            <span>No files ingested in this scope scope.</span>
          </div>
        ) : (
          <div className="max-h-60 overflow-y-auto space-y-1.5 pr-1">
            <AnimatePresence initial={false}>
              {filteredDocs.map((doc, idx) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.15 }}
                  className="flex items-center justify-between p-3 bg-[#171923]/40 border border-[#2a2f3d] rounded-lg hover:border-gray-500/40 transition-colors text-xs"
                >
                  <div className="flex items-center space-x-3 min-w-0">
                    <FileText className="h-4 w-4 text-[#8b5cf6] shrink-0" />
                    <span className="text-gray-200 font-medium truncate">{renderSafe(doc.filename)}</span>
                  </div>
                  <div className="flex items-center space-x-3 shrink-0">
                    <span className="text-[10px] text-gray-500">
                      {doc.upload_date ? new Date(doc.upload_date).toLocaleDateString() : 'N/A'}
                    </span>
                    <button
                      onClick={() => handleDelete(doc.filename)}
                      className="p-1 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors text-gray-500"
                      title="Remove file"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
