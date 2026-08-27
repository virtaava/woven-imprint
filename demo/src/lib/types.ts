export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface CharacterState {
  id: string
  name: string
  emotion: { mood: string; intensity: number; cause?: string }
  arc: { phase: string; tension: number }
}

export interface Memory {
  id: string
  content: string
  tier: string
  importance: number
  certainty?: number
  created_at?: string
  metadata?: { pinned?: boolean; fact_id?: string; historical?: boolean; [k: string]: unknown }
}

export interface Fact {
  id: string
  subject: string
  predicate: string
  object: string
  statement: string
  valid_from: string
  valid_to: string | null
  certainty: number
  memory_id?: string | null
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

export interface CharacterSummary {
  id: string
  character_id?: string
  name: string
}

export interface Session {
  id: string
  character_id: string
  alias: string | null
  summary: string | null
  started_at: string
  ended_at: string | null
}
