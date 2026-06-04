import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://rggffjpqtzdhjrsmwlyr.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

let supabaseClient: any;
let initError: string | null = null;

try {
  if (!supabaseAnonKey || supabaseAnonKey.trim() === '' || supabaseAnonKey === 'MISSING_ANON_KEY') {
    throw new Error('Supabase Anon Key is missing. Please configure VITE_SUPABASE_ANON_KEY in your .env file in the frontend folder.');
  }
  supabaseClient = createClient(supabaseUrl, supabaseAnonKey);
} catch (e: any) {
  console.error('Failed to initialize Supabase Client:', e);
  initError = e.message || String(e);
  
  // Safe dummy client to prevent runtime reference crashes during module import loading
  supabaseClient = {
    auth: {
      getSession: async () => ({ data: { session: null }, error: new Error(initError || '') }),
      onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
      signInWithPassword: async () => ({ error: new Error(initError || '') }),
      signUp: async () => ({ error: new Error(initError || '') }),
      signOut: async () => ({ error: null }),
    }
  };
}

export { initError };
export const supabase = supabaseClient;
