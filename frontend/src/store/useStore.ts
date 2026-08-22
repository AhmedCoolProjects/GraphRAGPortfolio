import { create } from 'zustand'

interface AppState {
  activeSection: string
  setActiveSection: (section: string) => void
  isDarkMode: boolean
  toggleDarkMode: () => void
  initialMessage: string | null
  setInitialMessage: (message: string | null) => void
}

export const useStore = create<AppState>((set) => ({
  activeSection: 'me',
  setActiveSection: (section) => set({ activeSection: section }),
  isDarkMode: false,
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
  initialMessage: null,
  setInitialMessage: (message) => set({ initialMessage: message }),
}))