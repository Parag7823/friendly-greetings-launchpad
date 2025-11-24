import { create } from 'zustand';

/**
 * File Processing Status Step
 * Tracks individual steps in the file processing pipeline
 */
export interface ProcessingStep {
  step: string;
  message: string;
  status: 'in_progress' | 'complete' | 'error';
  timestamp: number;
  progress?: number;
  extra?: Record<string, any>;
}

/**
 * File Status
 * Tracks the complete processing history for a single file
 */
export interface FileStatus {
  fileId: string;
  filename?: string;
  steps: ProcessingStep[];
  currentStep?: string;
  overallProgress: number;
  startedAt: number;
  completedAt?: number;
  error?: string;
}

/**
 * File Status Store
 * Global state management for file processing status
 * 
 * Features:
 * - Track multiple files simultaneously
 * - Store step history for each file
 * - Calculate overall progress
 * - Handle errors gracefully
 * - Auto-cleanup completed files
 */
interface FileStatusStore {
  statuses: Record<string, FileStatus>;
  activeFileId: string | null;
  
  // Actions
  addStep: (fileId: string, step: ProcessingStep) => void;
  updateProgress: (fileId: string, progress: number) => void;
  setError: (fileId: string, error: string) => void;
  markComplete: (fileId: string) => void;
  clearStatus: (fileId: string) => void;
  clearAll: () => void;
  setActiveFile: (fileId: string | null) => void;
  
  // Queries
  getStatus: (fileId: string) => FileStatus | undefined;
  getAllStatuses: () => Record<string, FileStatus>;
  getActiveFiles: () => FileStatus[];
  getCompletedFiles: () => FileStatus[];
}

export const useFileStatusStore = create<FileStatusStore>((set, get) => ({
  statuses: {},
  activeFileId: null,

  /**
   * Add a new processing step to a file's history
   */
  addStep: (fileId: string, step: ProcessingStep) =>
    set((state) => {
      const existing = state.statuses[fileId];
      const now = Date.now();

      return {
        statuses: {
          ...state.statuses,
          [fileId]: {
            fileId,
            filename: existing?.filename,
            steps: [...(existing?.steps || []), step],
            currentStep: step.step,
            overallProgress: step.progress || existing?.overallProgress || 0,
            startedAt: existing?.startedAt || now,
            completedAt: existing?.completedAt,
            error: existing?.error,
          },
        },
      };
    }),

  /**
   * Update overall progress for a file
   */
  updateProgress: (fileId: string, progress: number) =>
    set((state) => {
      const existing = state.statuses[fileId];
      if (!existing) return state;

      return {
        statuses: {
          ...state.statuses,
          [fileId]: {
            ...existing,
            overallProgress: Math.min(100, Math.max(0, progress)),
          },
        },
      };
    }),

  /**
   * Set error status for a file
   */
  setError: (fileId: string, error: string) =>
    set((state) => {
      const existing = state.statuses[fileId];
      if (!existing) return state;

      return {
        statuses: {
          ...state.statuses,
          [fileId]: {
            ...existing,
            error,
            overallProgress: 0,
            completedAt: Date.now(),
          },
        },
      };
    }),

  /**
   * Mark file processing as complete
   */
  markComplete: (fileId: string) =>
    set((state) => {
      const existing = state.statuses[fileId];
      if (!existing) return state;

      return {
        statuses: {
          ...state.statuses,
          [fileId]: {
            ...existing,
            overallProgress: 100,
            completedAt: Date.now(),
          },
        },
      };
    }),

  /**
   * Clear status for a specific file
   */
  clearStatus: (fileId: string) =>
    set((state) => {
      const { [fileId]: _, ...rest } = state.statuses;
      return { statuses: rest };
    }),

  /**
   * Clear all statuses
   */
  clearAll: () => set({ statuses: {}, activeFileId: null }),

  /**
   * Set active file for sticky panel (Step 5.1)
   * Panel stays open and updates content when activeFileId changes
   */
  setActiveFile: (fileId: string | null) =>
    set({ activeFileId: fileId }),

  /**
   * Get status for a specific file
   */
  getStatus: (fileId: string) => {
    return get().statuses[fileId];
  },

  /**
   * Get all statuses
   */
  getAllStatuses: () => {
    return get().statuses;
  },

  /**
   * Get all active (in-progress) files
   */
  getActiveFiles: () => {
    const statuses = get().statuses;
    return Object.values(statuses).filter(
      (status) => !status.completedAt && !status.error
    );
  },

  /**
   * Get all completed files
   */
  getCompletedFiles: () => {
    const statuses = get().statuses;
    return Object.values(statuses).filter(
      (status) => status.completedAt || status.error
    );
  },
}));
