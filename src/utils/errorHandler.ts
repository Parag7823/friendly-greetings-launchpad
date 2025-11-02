/**
 * Unified Error Handler
 * 
 * MISMATCH FIX #3: Provides consistent error handling across:
 * - Frontend toast notifications
 * - Backend HTTP exceptions
 * - WebSocket error messages
 */
import { config } from '@/config';
import type { ReactNode } from 'react';

// Define a minimal toast function signature locally to avoid importing React hooks here
type ToastFn = (args: {
  title?: ReactNode;
  description?: ReactNode;
  variant?: 'default' | 'destructive';
  duration?: number;
}) => unknown;

// Get toast function (will be set from component context)
let toastFn: ToastFn | null = null;

export const setToastFunction = (fn: ToastFn) => {
  toastFn = fn;
};

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum ErrorSource {
  FRONTEND = 'frontend',
  BACKEND = 'backend',
  WEBSOCKET = 'websocket',
  NETWORK = 'network',
  VALIDATION = 'validation'
}

export interface ErrorContext {
  message: string;
  severity: ErrorSeverity;
  source: ErrorSource;
  code?: string;
  details?: any;
  userId?: string;
  jobId?: string;
  retryable?: boolean;
}

export class UnifiedErrorHandler {
  /**
   * Handle error with unified strategy across all channels
   */
  static handle(error: ErrorContext): void {
    // Log to console for debugging
    console.error(`[${error.source}] ${error.severity.toUpperCase()}: ${error.message}`, error.details);

    // Show user-facing notification
    this.showUserNotification(error);

    // Track error for analytics (if needed)
    this.trackError(error);

    // Send to backend for logging (if critical)
    if (error.severity === ErrorSeverity.CRITICAL || error.severity === ErrorSeverity.HIGH) {
      this.reportToBackend(error);
    }
  }

  /**
   * Show user-facing toast notification
   */
  private static showUserNotification(error: ErrorContext): void {
    const title = this.getErrorTitle(error);
    const description = this.getUserFriendlyMessage(error);
    const variant = this.getToastVariant(error.severity);

    // Use toast function if available (set by component)
    if (toastFn) {
      toastFn({
        title,
        description,
        variant,
        duration: this.getToastDuration(error.severity)
      });
    } else {
      // Fallback to console if toast not available
      console.warn(`[${variant}] ${title}: ${description}`);
    }
  }

  /**
   * Get error title based on severity
   */
  private static getErrorTitle(error: ErrorContext): string {
    switch (error.severity) {
      case ErrorSeverity.CRITICAL:
        return '🔴 Critical Error';
      case ErrorSeverity.HIGH:
        return '⚠️ Error';
      case ErrorSeverity.MEDIUM:
        return '⚠️ Warning';
      case ErrorSeverity.LOW:
        return 'ℹ️ Notice';
      default:
        return 'Error';
    }
  }

  /**
   * Convert technical error to user-friendly message
   */
  private static getUserFriendlyMessage(error: ErrorContext): string {
    // Map common technical errors to user-friendly messages
    const errorMap: Record<string, string> = {
      'Network request failed': 'Unable to connect to server. Please check your internet connection.',
      'Session expired': 'Your session has expired. Please refresh the page.',
      'Unauthorized': 'You do not have permission to perform this action.',
      'File too large': 'The file is too large. Maximum size is 500MB.',
      'Invalid file type': 'Please upload a valid Excel or CSV file.',
      'Duplicate file detected': 'This file has already been uploaded.',
      'Processing failed': 'Unable to process the file. Please try again.',
      'WebSocket connection failed': 'Real-time updates unavailable. Using fallback mode.'
    };

    // Check if we have a friendly message
    for (const [key, value] of Object.entries(errorMap)) {
      if (error.message.includes(key)) {
        return value;
      }
    }

    // Return original message if no mapping found
    return error.message;
  }

  /**
   * Get toast variant based on severity
   */
  private static getToastVariant(severity: ErrorSeverity): 'default' | 'destructive' {
    return severity === ErrorSeverity.CRITICAL || severity === ErrorSeverity.HIGH
      ? 'destructive'
      : 'default';
  }

  /**
   * Get toast duration based on severity
   */
  private static getToastDuration(severity: ErrorSeverity): number {
    switch (severity) {
      case ErrorSeverity.CRITICAL:
        return 10000; // 10 seconds
      case ErrorSeverity.HIGH:
        return 7000;  // 7 seconds
      case ErrorSeverity.MEDIUM:
        return 5000;  // 5 seconds
      case ErrorSeverity.LOW:
        return 3000;  // 3 seconds
      default:
        return 5000;
    }
  }

  /**
   * Track error for analytics
   */
  private static trackError(error: ErrorContext): void {
    // TODO: Integrate with analytics service (e.g., Sentry, LogRocket)
    // For now, just log to console
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'exception', {
        description: error.message,
        fatal: error.severity === ErrorSeverity.CRITICAL,
        source: error.source
      });
    }
  }

  /**
   * Report critical errors to backend
   */
  private static async reportToBackend(error: ErrorContext): Promise<void> {
    try {
      // Only report in production
      if (process.env.NODE_ENV !== 'production') {
        return;
      }

      // Send error report to backend
      await fetch(`${config.apiUrl}/api/error-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...error,
          timestamp: new Date().toISOString(),
          userAgent: navigator.userAgent,
          url: window.location.href
        })
      }).catch(() => {
        // Silently fail if error reporting fails
        console.warn('Failed to report error to backend');
      });
    } catch (e) {
      // Silently fail
    }
  }

  /**
   * Parse backend HTTP error response
   */
  static parseBackendError(response: Response, defaultMessage: string = 'An error occurred'): ErrorContext {
    return {
      message: defaultMessage,
      severity: this.getSeverityFromStatus(response.status),
      source: ErrorSource.BACKEND,
      code: response.status.toString(),
      retryable: response.status >= 500
    };
  }

  /**
   * Parse WebSocket error
   */
  static parseWebSocketError(error: any): ErrorContext {
    return {
      message: error.message || 'WebSocket connection error',
      severity: ErrorSeverity.MEDIUM,
      source: ErrorSource.WEBSOCKET,
      retryable: true
    };
  }

  /**
   * Parse network error
   */
  static parseNetworkError(error: any): ErrorContext {
    return {
      message: error.message || 'Network error',
      severity: ErrorSeverity.HIGH,
      source: ErrorSource.NETWORK,
      retryable: true
    };
  }

  /**
   * Get severity from HTTP status code
   */
  private static getSeverityFromStatus(status: number): ErrorSeverity {
    if (status >= 500) return ErrorSeverity.CRITICAL;
    if (status >= 400) return ErrorSeverity.HIGH;
    if (status >= 300) return ErrorSeverity.MEDIUM;
    return ErrorSeverity.LOW;
  }

  /**
   * Create error context from exception
   */
  static fromException(error: Error, source: ErrorSource = ErrorSource.FRONTEND): ErrorContext {
    return {
      message: error.message,
      severity: ErrorSeverity.HIGH,
      source,
      details: {
        name: error.name,
        stack: error.stack
      },
      retryable: false
    };
  }

  /**
   * Create error context from React error
   */
  static fromReactError(error: Error, componentStack: ReactNode, source: ErrorSource = ErrorSource.FRONTEND): ErrorContext {
    return {
      message: error.message,
      severity: ErrorSeverity.HIGH,
      source,
      details: {
        name: error.name,
        stack: error.stack,
        componentStack: componentStack
      },
      retryable: false
    };
  }
}

// Export convenience functions
export const handleError = (error: ErrorContext) => UnifiedErrorHandler.handle(error);
export const handleBackendError = (response: Response, message?: string) => 
  UnifiedErrorHandler.handle(UnifiedErrorHandler.parseBackendError(response, message));
export const handleWebSocketError = (error: any) => 
  UnifiedErrorHandler.handle(UnifiedErrorHandler.parseWebSocketError(error));
export const handleNetworkError = (error: any) => 
  UnifiedErrorHandler.handle(UnifiedErrorHandler.parseNetworkError(error));
export const handleException = (error: Error, source?: ErrorSource) => 
  UnifiedErrorHandler.handle(UnifiedErrorHandler.fromException(error, source));
