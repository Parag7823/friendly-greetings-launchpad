/**
 * Unit Tests for AuthProvider Component
 * 
 * Tests:
 * - Session loading and state management
 * - Anonymous sign-in functionality
 * - Error handling for auth failures
 * - Auth state change listeners
 * - Context provider functionality
 */

import { render, screen, waitFor, act } from '@testing-library/react';
import { AuthProvider, useAuth } from '@/components/AuthProvider';
import { supabase } from '@/integrations/supabase/client';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

// Mock Supabase client
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    auth: {
      getSession: vi.fn(),
      signInAnonymously: vi.fn(),
      onAuthStateChange: vi.fn(),
    },
  },
}));

// Test component that uses the auth context
const TestComponent = () => {
  const { user, loading, signInAnonymously } = useAuth();
  
  return (
    <div>
      <div data-testid="loading">{loading ? 'loading' : 'loaded'}</div>
      <div data-testid="user">{user ? user.id : 'no-user'}</div>
      <button onClick={signInAnonymously} data-testid="sign-in-btn">
        Sign In
      </button>
    </div>
  );
};

describe('AuthProvider', () => {
  let mockSubscription: { unsubscribe: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    mockSubscription = { unsubscribe: vi.fn() };
    
    // Default mock implementation
    (supabase.auth.onAuthStateChange as any).mockReturnValue({
      data: { subscription: mockSubscription },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Session Loading', () => {
    it('should start in loading state', () => {
      (supabase.auth.getSession as any).mockReturnValue(
        new Promise(() => {}) // Never resolves
      );

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      expect(screen.getByTestId('loading')).toHaveTextContent('loading');
    });

    it('should load user session successfully', async () => {
      const mockUser = { id: 'user-123', email: 'test@example.com' };
      
      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: { user: mockUser } },
      });

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
        expect(screen.getByTestId('user')).toHaveTextContent('user-123');
      });
    });

    it('should handle null session', async () => {
      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: null },
      });

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
        expect(screen.getByTestId('user')).toHaveTextContent('no-user');
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle getSession error gracefully', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      (supabase.auth.getSession as any).mockRejectedValue(
        new Error('Supabase unavailable')
      );

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
        expect(screen.getByTestId('user')).toHaveTextContent('no-user');
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          'Failed to get auth session:',
          expect.any(Error)
        );
      });

      consoleErrorSpy.mockRestore();
    });

    it('should continue app functionality after auth error', async () => {
      (supabase.auth.getSession as any).mockRejectedValue(
        new Error('Network error')
      );

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
      });

      // App should still render and be functional
      expect(screen.getByTestId('sign-in-btn')).toBeInTheDocument();
    });
  });

  describe('Anonymous Sign-In', () => {
    it('should sign in anonymously successfully', async () => {
      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: null },
      });
      
      (supabase.auth.signInAnonymously as any).mockResolvedValue({
        data: { user: { id: 'anon-user-123' } },
        error: null,
      });

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
      });

      await act(async () => {
        screen.getByTestId('sign-in-btn').click();
      });

      expect(supabase.auth.signInAnonymously).toHaveBeenCalled();
    });

    it('should handle sign-in error', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: null },
      });
      
      const signInError = new Error('Sign-in failed');
      (supabase.auth.signInAnonymously as any).mockResolvedValue({
        data: null,
        error: signInError,
      });

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
      });

      await expect(async () => {
        await act(async () => {
          screen.getByTestId('sign-in-btn').click();
        });
      }).rejects.toThrow('Sign-in failed');

      consoleErrorSpy.mockRestore();
    });
  });

  describe('Auth State Changes', () => {
    it('should listen to auth state changes', async () => {
      let authCallback: ((event: string, session: any) => void) | null = null;

      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: null },
      });

      (supabase.auth.onAuthStateChange as any).mockImplementation((callback) => {
        authCallback = callback;
        return { data: { subscription: mockSubscription } };
      });

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('user')).toHaveTextContent('no-user');
      });

      // Simulate auth state change
      const newUser = { id: 'new-user-456' };
      act(() => {
        authCallback?.('SIGNED_IN', { user: newUser });
      });

      await waitFor(() => {
        expect(screen.getByTestId('user')).toHaveTextContent('new-user-456');
      });
    });

    it('should unsubscribe on unmount', async () => {
      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: null },
      });

      const { unmount } = render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
      });

      unmount();

      expect(mockSubscription.unsubscribe).toHaveBeenCalled();
    });
  });

  describe('Context Provider', () => {
    it('should throw error when useAuth used outside provider', () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        render(<TestComponent />);
      }).toThrow('useAuth must be used within an AuthProvider');

      consoleErrorSpy.mockRestore();
    });

    it('should provide auth context to children', async () => {
      const mockUser = { id: 'user-789' };
      
      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: { user: mockUser } },
      });

      render(
        <AuthProvider>
          <TestComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('user')).toHaveTextContent('user-789');
      });
    });
  });

  describe('Performance', () => {
    it('should not re-render unnecessarily', async () => {
      let renderCount = 0;
      
      const CountingComponent = () => {
        renderCount++;
        const { user } = useAuth();
        return <div>{user?.id || 'no-user'}</div>;
      };

      (supabase.auth.getSession as any).mockResolvedValue({
        data: { session: { user: { id: 'user-123' } } },
      });

      render(
        <AuthProvider>
          <CountingComponent />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByText('user-123')).toBeInTheDocument();
      });

      // Should render exactly twice: initial + after session loaded
      expect(renderCount).toBe(2);
    });
  });
});
