export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface CharacterState {
  id: string
  name: string
  emotion: { mood: string; intensity: number; cause?: string }
  arc: { phase: string; tension: number }
}

export interface Memory {
  content: string
  tier: string
  importance: number
  created_at?: string
}

export interface Relationship {
  trust: number
  affection: number
  respect: number
  familiarity: number
  tension: number
}

export interface ProviderConfig {
  provider: string
  model: string
  base_url: string | null
  api_key_configured: boolean
}
