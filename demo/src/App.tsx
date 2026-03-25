import { useState, useEffect, useCallback } from 'react'
import { TopBar } from '@/components/TopBar'
import { ChatPanel } from '@/components/ChatPanel'
import { XRayPanel } from '@/components/XRayPanel'
import { ProviderModal } from '@/components/ProviderModal'
import { CharacterDrawer } from '@/components/CharacterDrawer'
import {
  getProviderConfig,
  testProviderConnection,
  fetchCharacters,
  startSession,
  endSession,
  sendMessage,
  fetchCharacterState,
  recallMemories,
  fetchRelationship,
  reflectCharacter,
} from '@/lib/api'
import type { ChatMessage, CharacterState, CharacterSummary, Memory, Relationship, ProviderConfig } from '@/lib/types'

const MERIDIAN_GREETING: ChatMessage = {
  role: 'assistant',
  content:
    "Ah \u2014 a new visitor. Welcome. I am **Meridian**, Keeper of the Imprint. I remember everything that matters about the people I meet. Your name, what you're building, the questions that keep you up at night. That is what this place does \u2014 it gives characters like me a real memory.\n\nWhat shall I call you?",
}

const SETUP_PROMPT: ChatMessage = {
  role: 'assistant',
  content:
    "Welcome to the **woven-imprint** demo. Before we begin, I need an LLM provider to power my responses.\n\nClick the **\u2699 Settings** button in the top bar to configure your provider (OpenAI, Anthropic, DeepSeek, or a local Ollama model). Once that's set, we can talk properly.\n\nThe **X-Ray panel** on the right will show my memory, emotions, and relationships updating in real time as we chat.",
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([SETUP_PROMPT])
  const [loading, setLoading] = useState(false)
  const [reflectLoading, setReflectLoading] = useState(false)
  const [characterState, setCharacterState] = useState<CharacterState | null>(null)
  const [memories, setMemories] = useState<Memory[]>([])
  const [relationship, setRelationship] = useState<Relationship | null>(null)
  const [providerConfig, setProviderConfig] = useState<ProviderConfig | null>(null)
  const [showProviderModal, setShowProviderModal] = useState(false)
  const [showCharacterDrawer, setShowCharacterDrawer] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(() =>
    localStorage.getItem('woven-session-id')
  )
  const [characterId, setCharacterId] = useState<string | null>(() =>
    localStorage.getItem('woven-character-id')
  )
  const [characters, setCharacters] = useState<CharacterSummary[]>([])
  const [searchResults, setSearchResults] = useState<Memory[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [xrayVisible, setXrayVisible] = useState(() => {
    const saved = localStorage.getItem('woven-xray-visible')
    return saved !== null ? saved === 'true' : true  // default visible
  })

  // Persist UI state to localStorage
  useEffect(() => {
    localStorage.setItem('woven-xray-visible', String(xrayVisible))
  }, [xrayVisible])
  useEffect(() => {
    if (characterId) localStorage.setItem('woven-character-id', characterId)
    else localStorage.removeItem('woven-character-id')
  }, [characterId])
  useEffect(() => {
    if (sessionId) localStorage.setItem('woven-session-id', sessionId)
    else localStorage.removeItem('woven-session-id')
  }, [sessionId])

  // Load provider config on mount — always verify connection works
  useEffect(() => {
    async function init() {
      try {
        const config = await getProviderConfig()
        setProviderConfig(config)

        // Test the actual connection before assuming it works
        if (config.api_key_configured || config.provider === 'ollama') {
          const test = await testProviderConnection({
            provider: config.provider,
            model: config.model,
          })
          if (test.success) {
            setMessages([MERIDIAN_GREETING])
            await initCharacter()
            return
          }
        }
        // Connection failed or no key — setup prompt is already showing,
        // let the user read it and open Settings manually
      } catch {
        // API not available — setup prompt already visible
      }
    }
    init()
  }, [])

  const refreshCharacters = useCallback(async () => {
    try {
      const chars = await fetchCharacters()
      const charList = chars.characters || chars
      if (Array.isArray(charList)) {
        setCharacters(charList.map((c: any) => ({ id: c.id || c.character_id, name: c.name })))
      }
    } catch {
      // ignore
    }
  }, [])

  const initCharacter = useCallback(async () => {
    try {
      const chars = await fetchCharacters()
      const charList = chars.characters || chars
      if (Array.isArray(charList)) {
        setCharacters(charList.map((c: any) => ({ id: c.id || c.character_id, name: c.name })))
      }
      if (!Array.isArray(charList) || charList.length === 0) return

      // Resume stored character if still available, otherwise pick the first
      const storedId = localStorage.getItem('woven-character-id')
      const storedSession = localStorage.getItem('woven-session-id')
      const match = storedId && charList.find((c: any) => (c.id || c.character_id) === storedId)
      const char = match || charList[0]
      const id = char.id || char.character_id
      setCharacterId(id)

      // If resuming a stored session, skip starting a new one — the server
      // will auto-start on the first chat if needed. Only start a fresh
      // session when there's no stored session for this character.
      if (match && storedSession) {
        setSessionId(storedSession)
      } else {
        const session = await startSession(id)
        setSessionId(session.session_id || session.id || 'active')
      }

      const state = await fetchCharacterState(id)
      setCharacterState(state)

      try {
        const mems = await recallMemories(id, 'recent', 10)
        setMemories(Array.isArray(mems) ? mems : mems.memories || [])
      } catch {
        // No memories yet
      }
      try {
        const rel = await fetchRelationship(id, 'user')
        setRelationship(rel)
      } catch {
        // No relationship yet
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
        const model = 'meridian'  // Demo server matches character by name
        const allMessages = [...messages, userMsg].map((m) => ({
          role: m.role,
          content: m.content,
        }))

        const res = await sendMessage(model, allMessages)
        const content =
          res.choices?.[0]?.message?.content || res.content || res.response || 'Something went wrong \u2014 check your provider configuration in Settings.'

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

  const handleReflect = useCallback(
    async () => {
      if (!characterId) return
      setReflectLoading(true)

      try {
        const res = await reflectCharacter(characterId)
        const reflection = res.reflection || res.result || 'Internal reflection completed.'

        const systemMsg: ChatMessage = {
          role: 'system',
          content: `✦ ${reflection}`,
        }
        setMessages((prev) => [...prev, systemMsg])

        // Refresh X-Ray after reflection
        await refreshXRay()
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'system',
            content: 'Failed to reflect. Please try again.',
          },
        ])
      } finally {
        setReflectLoading(false)
      }
    },
    [characterId, refreshXRay]
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
      setMessages([MERIDIAN_GREETING])
      if (!characterId) {
        await initCharacter()
      }
    },
    [characterId, initCharacter]
  )

  const switchCharacter = useCallback(
    async (newId: string) => {
      // End current session if active
      if (characterId && sessionId) {
        try {
          await endSession(characterId)
        } catch {
          // ignore — session may already be ended
        }
      }

      setCharacterId(newId)

      // Start new session
      try {
        const session = await startSession(newId)
        setSessionId(session.session_id || session.id || 'active')
      } catch {
        setSessionId(null)
      }

      // Load character state
      try {
        const state = await fetchCharacterState(newId)
        setCharacterState(state)

        // Reset messages with character-specific greeting
        const greeting: ChatMessage = {
          role: 'assistant',
          content: `Hello! I am **${state.name}**. How can I help you today?`,
        }
        setMessages([greeting])
      } catch {
        setMessages([MERIDIAN_GREETING])
      }

      // Refresh X-Ray data
      setMemories([])
      setRelationship(null)
      setSearchResults([])
      try {
        const mems = await recallMemories(newId, 'recent', 10)
        setMemories(Array.isArray(mems) ? mems : mems.memories || [])
      } catch {
        // No memories yet
      }
      try {
        const rel = await fetchRelationship(newId, 'user')
        setRelationship(rel)
      } catch {
        // No relationship yet
      }
    },
    [characterId, sessionId]
  )

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <TopBar
        character={characterState}
        provider={providerConfig}
        sessionId={sessionId}
        memoryCount={memories.length}
        onOpenProviderModal={() => setShowProviderModal(true)}
        onOpenCharacterDrawer={() => setShowCharacterDrawer(true)}
        xrayVisible={xrayVisible}
        onToggleXray={() => setXrayVisible(v => !v)}
      />

      <div className="flex flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Chat panel */}
        <div
          className={`flex min-h-0 flex-col transition-all duration-300 ${
            xrayVisible ? 'w-full lg:w-[70%]' : 'w-full'
          }`}
        >
          <ChatPanel
            messages={messages}
            loading={loading}
            onSend={handleSend}
            onReflect={handleReflect}
            reflectLoading={reflectLoading}
            characterSelected={!!characterId}
          />
        </div>

        {/* X-Ray panel */}
        {xrayVisible && (
          <div className="min-h-0 w-full border-t border-border/60 transition-all duration-300 lg:w-[30%] lg:border-l lg:border-t-0">
            <XRayPanel
              character={characterState}
              memories={memories}
              relationship={relationship}
              onSearchMemory={handleSearchMemory}
              searchResults={searchResults}
              searchLoading={searchLoading}
            />
          </div>
        )}
      </div>

      <ProviderModal
        open={showProviderModal}
        onOpenChange={setShowProviderModal}
        currentConfig={providerConfig}
        onSaved={handleProviderSaved}
      />

      <CharacterDrawer
        open={showCharacterDrawer}
        onOpenChange={setShowCharacterDrawer}
        characters={characters}
        activeCharacterId={characterId}
        onSelectCharacter={switchCharacter}
        onRefreshCharacters={refreshCharacters}
      />
    </div>
  )
}
