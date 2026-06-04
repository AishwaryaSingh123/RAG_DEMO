import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { supabase, initError } from './lib/supabase';
import type { Session } from '@supabase/supabase-js';
import { AlertCircle, Terminal, HelpCircle } from 'lucide-react';

import Login from './pages/Login';
import Register from './pages/Register';
import Chat from './pages/Chat';
import UserManagement from './pages/UserManagement';
import KnowledgeBases from './pages/KnowledgeBases';

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (initError) {
      setLoading(false);
      return;
    }

    supabase.auth.getSession().then((res: any) => {
      setSession((res?.data?.session || null) as Session | null);
      setLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event: any, session: any) => {
      setSession(session as Session | null);
    });

    return () => subscription.unsubscribe();
  }, []);

  // 1. Render Supabase Configuration Error State (Premium design)
  if (initError) {
    return (
      <div className="min-h-screen bg-[#0f1117] flex items-center justify-center p-4 md:p-8 font-sans">
        <div className="max-w-xl w-full bg-[#1e2230] border border-[#ef4444]/30 rounded-2xl p-6 md:p-8 shadow-2xl relative overflow-hidden">
          {/* Subtle top glow */}
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 to-amber-500" />
          
          <div className="flex items-center space-x-4 mb-6">
            <div className="h-12 w-12 rounded-xl bg-red-500/10 flex items-center justify-center text-red-500 shrink-0">
              <AlertCircle className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">Configuration Error</h1>
              <p className="text-xs text-gray-400">Failed to initialize core authentication system</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-[#1a1f2e] border border-[#2a2f3d] rounded-xl p-4">
              <div className="flex items-center space-x-2 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                <Terminal className="h-3.5 w-3.5 text-purple-500" />
                <span>Error details</span>
              </div>
              <p className="text-sm text-red-400 font-mono leading-relaxed break-words">
                {initError}
              </p>
            </div>

            <div className="text-sm text-gray-300 space-y-3 leading-relaxed">
              <p>
                To resolve this issue, configure your environment file:
              </p>
              <ol className="list-decimal list-inside space-y-2 text-gray-400 pl-1 text-xs">
                <li>Create/edit the file <code className="bg-[#1a1f2e] text-[#7c3aed] px-1.5 py-0.5 rounded font-mono">.env</code> inside the <code className="bg-[#1a1f2e] text-[#f3f4f6] px-1.5 py-0.5 rounded font-mono">frontend/</code> directory.</li>
                <li>Add your Supabase Anon Key:
                  <pre className="bg-[#1a1f2e] text-[#a78bfa] p-3 rounded-lg mt-2 font-mono text-xs overflow-x-auto border border-[#2a2f3d]">
VITE_SUPABASE_ANON_KEY=your_actual_supabase_anon_key</pre>
                </li>
                <li>Save the file and restart the development server.</li>
              </ol>
            </div>

            <div className="border-t border-[#2a2f3d] pt-4 mt-6 flex justify-between items-center text-xs text-gray-500">
              <span className="flex items-center">
                <HelpCircle className="h-3.5 w-3.5 mr-1" />
                Need help? Check docs
              </span>
              <span>RAG SaaS platform</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 2. Auth checking state (Premium design loading spinner)
  if (loading) {
    return (
      <div className="min-h-screen bg-[#0f1117] flex flex-col items-center justify-center font-sans">
        <div className="relative flex items-center justify-center">
          {/* Pulsing glow background */}
          <div className="absolute h-16 w-16 rounded-full bg-[#7c3aed]/20 blur-xl animate-pulse" />
          {/* Animated spinner ring */}
          <div className="h-12 w-12 rounded-full border-2 border-gray-800 border-t-2 border-t-[#7c3aed] animate-spin" />
        </div>
        <p className="mt-4 text-xs font-medium text-gray-400 uppercase tracking-widest animate-pulse">
          Securing session...
        </p>
      </div>
    );
  }

  // 3. Main router entry
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={!session ? <Login /> : <Navigate to="/" />} />
        <Route path="/register" element={!session ? <Register /> : <Navigate to="/" />} />
        {/* Protected Routes */}
        <Route path="/" element={session ? <Chat /> : <Navigate to="/login" />} />
        <Route path="/users" element={session ? <UserManagement /> : <Navigate to="/login" />} />
        <Route path="/kbs" element={session ? <KnowledgeBases /> : <Navigate to="/login" />} />
      </Routes>
    </BrowserRouter>
  );
}
