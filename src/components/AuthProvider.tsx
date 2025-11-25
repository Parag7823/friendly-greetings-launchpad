import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { User } from '@supabase/supabase-js';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signInAnonymously: () => Promise<void>;
  getToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // FIX #1: Auto sign-in anonymously if no session exists
    supabase.auth.getSession()
      .then(({ data: { session } }) => {
        if (!session) {
          // Auto sign-in anonymously for better UX
          console.log('No session found, signing in anonymously...');
          return supabase.auth.signInAnonymously();
        }
        setUser(session?.user ?? null);
        setLoading(false);
      })
      .then((result) => {
        if (result && 'data' in result) {
          setUser(result.data.user ?? null);
        }
        setLoading(false);
      })
      .catch((error) => {
        console.error('Failed to get auth session:', error);
        setUser(null);
        setLoading(false);
        // Allow app to continue even if auth fails
      });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signInAnonymously = async () => {
    const { error } = await supabase.auth.signInAnonymously();
    if (error) {
      console.error('Error signing in anonymously:', error);
      throw error;
    }
  };

  // FIX #1: Add getToken method to retrieve JWT token for API requests
  const getToken = async (): Promise<string | null> => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      return session?.access_token ?? null;
    } catch (error) {
      console.error('Failed to get auth token:', error);
      return null;
    }
  };

  const value = {
    user,
    loading,
    signInAnonymously,
    getToken,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};