import { useState } from 'react';
import type { FormEvent } from 'react';
import { supabase } from '../lib/supabase';
import { Link, useNavigate } from 'react-router-dom';
import { Bot, Mail, Lock, Eye, EyeOff, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleRegister = async (e: FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      setError('Please fill in all fields.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
      });

      if (error) {
        setError(error.message);
      } else {
        setSuccess('Registration successful! Please check your email to confirm your account.');
        setTimeout(() => {
          navigate('/login');
        }, 4000);
      }
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f1117] flex items-center justify-center p-4 relative overflow-hidden font-sans">
      {/* Dynamic Background Glows */}
      <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-[#7c3aed]/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-500/10 blur-[120px] pointer-events-none" />

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="w-full max-w-md bg-[#1e2230]/70 backdrop-blur-xl border border-[#2a2f3d] rounded-2xl p-8 shadow-2xl relative z-10"
      >
        {/* Brand Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center h-14 w-14 rounded-2xl bg-gradient-to-tr from-[#7c3aed] to-blue-500 p-0.5 shadow-lg shadow-[#7c3aed]/20 mb-4">
            <div className="h-full w-full bg-[#171923] rounded-[14px] flex items-center justify-center text-white">
              <Bot className="h-7 w-7 text-[#8b5cf6]" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Create Account</h2>
          <p className="text-xs text-gray-400 mt-1">Get started with your RAG Research Workspace</p>
        </div>

        <form className="space-y-5" onSubmit={handleRegister}>
          {error && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-red-500/10 border border-red-500/30 text-red-400 p-3 rounded-xl text-xs flex items-start space-x-2"
            >
              <AlertCircle className="h-4 w-4 shrink-0 text-red-500" />
              <span>{error}</span>
            </motion.div>
          )}

          {success && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-green-500/10 border border-green-500/30 text-green-400 p-3 rounded-xl text-xs flex items-start space-x-2"
            >
              <div className="h-4 w-4 rounded-full bg-green-500/20 text-green-500 flex items-center justify-center shrink-0">✓</div>
              <span>{success}</span>
            </motion.div>
          )}

          {/* Email input field */}
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Email Address
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 flex items-center pl-3.5 pointer-events-none text-gray-500">
                <Mail className="h-4 w-4" />
              </span>
              <input
                type="email"
                required
                placeholder="you@example.com"
                className="w-full pl-10 pr-4 py-2.5 bg-[#1a1f2e] border border-[#2a2f3d] rounded-xl text-sm text-white placeholder-gray-500 outline-none transition-all duration-200 focus:border-[#7c3aed] focus:ring-2 focus:ring-[#7c3aed]/20"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
          </div>

          {/* Password input field */}
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Password
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 flex items-center pl-3.5 pointer-events-none text-gray-500">
                <Lock className="h-4 w-4" />
              </span>
              <input
                type={showPassword ? 'text' : 'password'}
                required
                placeholder="Minimum 6 characters"
                className="w-full pl-10 pr-11 py-2.5 bg-[#1a1f2e] border border-[#2a2f3d] rounded-xl text-sm text-white placeholder-gray-500 outline-none transition-all duration-200 focus:border-[#7c3aed] focus:ring-2 focus:ring-[#7c3aed]/20"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <button
                type="button"
                className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-gray-500 hover:text-gray-300 transition-colors"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Submit action */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center py-2.5 px-4 bg-gradient-to-r from-[#7c3aed] to-blue-600 hover:from-[#8b5cf6] hover:to-blue-500 text-white rounded-xl font-medium text-sm transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#7c3aed] disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-[#7c3aed]/20"
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                  <span>Registering...</span>
                </div>
              ) : (
                'Register'
              )}
            </button>
          </div>
        </form>

        <div className="mt-8 text-center text-xs">
          <span className="text-gray-400">Already have an account? </span>
          <Link to="/login" className="text-[#8b5cf6] font-semibold hover:text-[#a78bfa] transition-colors">
            Sign In
          </Link>
        </div>
      </motion.div>
    </div>
  );
}
