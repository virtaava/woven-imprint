import { useState, useEffect, useCallback } from 'react'
import { TopBar } from '@/components/TopBar'
import { ChatPanel } from '@/components/ChatPanel'
import { XRayPanel } from '@/components/XRayPanel'
import { ProviderModal } from '@/components/ProviderModal'
import {
  getProviderConfig,
  fetchCharacters,
  startSession,
  sendMessage,
  fetchCharacterState,
  recallMemories,
  fetchRelationship,
} from '@/lib/api'
import type { ChatMessage, CharacterState, Memory, Relationship, ProviderConfig } from '@/lib/types'

const MERIDIAN_GREETING: ChatMessage = {
  role: 'assistant',
  content:
    "Hello! I'm **Meridian**, a character powered by [woven-imprint](https://github.com/virtaava/woven-imprint). I have persistent memory, evolving emotions, and relationships that develop over time.\n\nAsk me anything about how woven-imprint works, or just chat \u2014 and watch the X-Ray panel on the right to see my internal state update in real time.",
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([MERIDIAN_GREETING])
  const [loading, setLoading] = useState(false)
  const [characterState, setCharacterState] = useState<CharacterState | null>(null)
  const [memories, setMemories] = useState<Memory[]>([])
  const [relationship, setRelationship] = useState<Relationship | null>(null)
  const [providerConfig, setProviderConfig] = useState<ProviderConfig | null>(null)
  const [showProviderModal, setShowProviderModal] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [characterId, setCharacterId] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<Memory[]>([])
  const [searchLoading, setSearchLoading] = useState(false)

  // Load provider config on mount
  useEffect(() => {
    async function init() {
      try {
        const config = await getProviderConfig()
        setProviderConfig(config)
        if (!config.api_key_configured && config.provider !== 'ollama') {
          setShowProviderModal(true)
        } else {
          await initCharacter()
        }
      } catch {
        // API not available, show modal
        setShowProviderModal(true)
      }
    }
    init()
  }, [])

  const initCharacter = useCallback(async () => {
    try {
      const chars = await fetchCharacters()
      const charList = chars.characters || chars
      if (Array.isArray(charList) && charList.length > 0) {
        const char = charList[0]
        const id = char.id || char.character_id
        setCharacterId(id)

        const session = await startSession(id)
        setSessionId(session.session_id || session.id || 'active')

        const state = await fetchCharacterState(id)
        setCharacterState(state)

        // Try to load initial memories
        try {
          const mems = await recallMemories(id, 'recent', 10)
          setMemories(Array.isArray(mems) ? mems : mems.memories || [])
        } catch {
          // No memories yet
        }

        // Try to load relationship
        try {
          const rel = await fetchRelationship(id, 'user')
          setRelationship(rel)
        } catch {
          // No relationship yet
        }
      }
    } catch {
      // Characters not available yet
    }
  }, [])

  const refreshXRay = useCallback(async () => {
    if (!characterId) return
    try {
      const state = await fetchCharacterState(characterId)
      setCharacterState(state)
    } catch {
      // ignore
    }
    try {
      const mems = await recallMemories(characterId, 'recent', 10)
      setMemories(Array.isArray(mems) ? mems : mems.memories || [])
    } catch {
      // ignore
    }
    try {
      const rel = await fetchRelationship(characterId, 'user')
      setRelationship(rel)
    } catch {
      // ignore
    }
  }, [characterId])

  const handleSend = useCallback(
    async (text: string) => {
      const userMsg: ChatMessage = { role: 'user', content: text }
      setMessages((prev) => [...prev, userMsg])
      setLoading(true)

      try {
        const model = providerConfig?.model || 'gpt-4o-mini'
        const allMessages = [...messages, userMsg].map((m) => ({
          role: m.role,
          content: m.content,
        }))

        const res = await sendMessage(model, allMessages)
        const content =
          res.choices?.[0]?.message?.content || res.content || res.response || 'I seem to have lost my train of thought.'

        const assistantMsg: ChatMessage = { role: 'assistant', content }
        setMessages((prev) => [...prev, assistantMsg])

        // Refresh X-Ray after response
        await refreshXRay()
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: 'Something went wrong while processing your message. Please check the provider configuration.',
          },
        ])
      } finally {
        setLoading(false)
      }
    },
    [messages, providerConfig, refreshXRay]
  )

  const handleSearchMemory = useCallback(
    async (query: string) => {
      if (!characterId) return
      setSearchLoading(true)
      try {
        const res = await recallMemories(characterId, query, 5)
        setSearchResults(Array.isArray(res) ? res : res.memories || [])
      } catch {
        setSearchResults([])
      } finally {
        setSearchLoading(false)
      }
    },
    [characterId]
  )

  const handleProviderSaved = useCallback(
    async (config: ProviderConfig) => {
      setProviderConfig(config)
      if (!characterId) {
        await initCharacter()
      }
    },
    [characterId, initCharacter]
  )

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <TopBar
        character={characterState}
        provider={providerConfig}
        sessionId={sessionId}
        memoryCount={memories.length}
        onOpenProviderModal={() => setShowProviderModal(true)}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Chat panel - 70% */}
        <div className="flex w-[70%] flex-col">
          <ChatPanel messages={messages} loading={loading} onSend={handleSend} />
        </div>

        {/* X-Ray panel - 30% */}
        <div className="w-[30%]">
          <XRayPanel
            character={characterState}
            memories={memories}
            relationship={relationship}
            onSearchMemory={handleSearchMemory}
            searchResults={searchResults}
            searchLoading={searchLoading}
          />
        </div>
      </div>

      <ProviderModal
        open={showProviderModal}
        onOpenChange={setShowProviderModal}
        currentConfig={providerConfig}
        onSaved={handleProviderSaved}
      />
    </div>
  )
}
