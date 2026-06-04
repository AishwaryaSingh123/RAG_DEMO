import { useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';
import { 
  LogOut, Users, RefreshCw, ArrowLeft, ShieldAlert,
  Shield, UserCheck, Search, Filter, AlertCircle
} from 'lucide-react';
import { motion } from 'framer-motion';

export default function UserManagement() {
  const navigate = useNavigate();
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  const fetchUsers = async () => {
    setLoading(true);
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;

    try {
      const res = await fetch('/api/users', {
        headers: {
          Authorization: `Bearer ${session.access_token}`,
        }
      });
      if (res.ok) {
        const data = await res.json();
        setUsers(data);
      }
    } catch (err) {
      console.error('Failed to fetch users:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();

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

  // Filter users based on search
  const filteredUsers = users.filter(u => 
    u.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    u.role?.toLowerCase().includes(searchTerm.toLowerCase())
  );

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
            <button onClick={() => navigate('/kbs')} className="text-gray-400 hover:text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-colors">Knowledge Bases</button>
            <button className="bg-[#252a3a] text-white px-3 py-1.5 rounded-lg text-sm font-medium border border-[#2a2f3d]">Users</button>
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

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 space-y-8 z-10">
        {/* Banner Section */}
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center tracking-tight">
              <Users className="h-6 w-6 mr-3 text-[#8b5cf6]" />
              User Management
            </h2>
            <p className="text-xs text-gray-400 mt-1">Review tenant members, allocate database roles, and administer security access controls.</p>
          </div>
          <div className="flex items-center space-x-2">
            <button 
              onClick={fetchUsers}
              title="Refresh list"
              className="p-2 bg-[#1e2230] border border-[#2a2f3d] hover:border-gray-500 rounded-lg text-gray-400 hover:text-white transition-all"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Top KPI Summaries */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-[#1e2230]/50 backdrop-blur border border-[#2a2f3d] rounded-xl p-5 flex items-center space-x-4">
            <div className="h-10 w-10 bg-purple-500/10 rounded-lg flex items-center justify-center text-[#8b5cf6]">
              <Users className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">{users.length}</div>
              <div className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">Tenant Members</div>
            </div>
          </div>
          <div className="bg-[#1e2230]/50 backdrop-blur border border-[#2a2f3d] rounded-xl p-5 flex items-center space-x-4">
            <div className="h-10 w-10 bg-blue-500/10 rounded-lg flex items-center justify-center text-blue-400">
              <Shield className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">
                {users.filter(u => u.role?.toLowerCase() === 'admin').length}
              </div>
              <div className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">Admins</div>
            </div>
          </div>
          <div className="bg-[#1e2230]/50 backdrop-blur border border-[#2a2f3d] rounded-xl p-5 flex items-center space-x-4">
            <div className="h-10 w-10 bg-green-500/10 rounded-lg flex items-center justify-center text-green-400">
              <UserCheck className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">
                {users.length > 0 ? users.length : 0}
              </div>
              <div className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">Verified Logins</div>
            </div>
          </div>
        </div>

        {/* Table & Controls Section */}
        <div className="bg-[#1e2230]/50 backdrop-blur border border-[#2a2f3d] rounded-xl overflow-hidden shadow-xl">
          {/* Table Headers / Filters */}
          <div className="p-4 bg-[#171923]/40 border-b border-[#2a2f3d] flex flex-col sm:flex-row justify-between items-center gap-3">
            <div className="relative w-full sm:max-w-xs">
              <Search className="h-4 w-4 text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search by email or role..."
                className="w-full pl-9 pr-4 py-1.5 bg-[#1a1f2e] border border-[#2a2f3d] rounded-lg text-xs text-white placeholder-gray-500 outline-none transition-all focus:border-[#7c3aed]"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="flex items-center space-x-2 text-xs text-gray-500">
              <Filter className="h-3.5 w-3.5" />
              <span>Displaying {filteredUsers.length} records</span>
            </div>
          </div>

          {/* Table Content */}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-[#2a2f3d]">
              <thead className="bg-[#171923]/70 text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                <tr>
                  <th scope="col" className="px-6 py-3.5 text-left font-semibold">User Identity</th>
                  <th scope="col" className="px-6 py-3.5 text-left font-semibold">Allocated Role</th>
                  <th scope="col" className="px-6 py-3.5 text-left font-semibold">Security Level</th>
                  <th scope="col" className="px-6 py-3.5 text-left font-semibold">Access Privilege</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#2a2f3d] bg-[#1e2230]/20 text-xs">
                {loading ? (
                  [1, 2, 3].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-1/2" /></td>
                      <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-1/3" /></td>
                      <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-1/4" /></td>
                      <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-1/5" /></td>
                    </tr>
                  ))
                ) : filteredUsers.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-6 py-12 text-center text-gray-500 font-medium">
                      <div className="max-w-xs mx-auto space-y-2">
                        <AlertCircle className="h-8 w-8 text-gray-600 mx-auto opacity-60" />
                        <h4 className="text-white text-sm">No Records Found</h4>
                        <p className="text-[11px] text-gray-500 leading-relaxed">
                          We couldn't retrieve any tenant users. Make sure you are authenticated with correct admin privileges.
                        </p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  filteredUsers.map((u, i) => {
                    const isAdmin = u.role?.toLowerCase() === 'admin';
                    return (
                      <tr key={i} className="hover:bg-[#252a3a]/30 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center space-x-3">
                            <div className="h-7 w-7 rounded-full bg-gradient-to-tr from-[#7c3aed] to-blue-500 flex items-center justify-center text-[10px] font-bold text-white shadow-sm">
                              {u.email ? u.email.charAt(0).toUpperCase() : 'U'}
                            </div>
                            <div className="font-medium text-gray-200">{u.email}</div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wide ${
                            isAdmin 
                              ? 'bg-purple-500/10 border border-purple-500/20 text-[#a78bfa]' 
                              : 'bg-blue-500/10 border border-blue-500/20 text-blue-400'
                          }`}>
                            {isAdmin ? 'Administrator' : 'Standard User'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-gray-400 font-mono text-[10px]">
                          <div className="flex items-center space-x-1.5">
                            {isAdmin ? (
                              <>
                                <ShieldAlert className="h-3.5 w-3.5 text-[#8b5cf6]" />
                                <span>Full Root privileges</span>
                              </>
                            ) : (
                              <>
                                <Shield className="h-3.5 w-3.5 text-blue-400" />
                                <span>Scoped user space</span>
                              </>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="inline-flex items-center text-[10px] font-semibold text-green-400 bg-green-500/5 border border-green-500/10 px-2 py-0.5 rounded-full">
                            Active
                          </span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
