import { useState, useEffect, useCallback } from 'react'
import { TopBar } from '@/components/TopBar'
import { ChatPanel } from '@/components/ChatPanel'
import { XRayPanel } from '@/components/XRayPanel'
import { ProviderModal } from '@/components/ProviderModal'
import {
  getProviderConfig,
  testProviderConnection,
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
  const [characterState, setCharacterState] = useState<CharacterState | null>(null)
  const [memories, setMemories] = useState<Memory[]>([])
  const [relationship, setRelationship] = useState<Relationship | null>(null)
  const [providerConfig, setProviderConfig] = useState<ProviderConfig | null>(null)
  const [showProviderModal, setShowProviderModal] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [characterId, setCharacterId] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<Memory[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [xrayVisible, setXrayVisible] = useState(() => {
    const saved = localStorage.getItem('woven-xray-visible')
    return saved !== null ? saved === 'true' : true  // default visible
  })

  // Persist X-Ray visibility to localStorage
  useEffect(() => {
    localStorage.setItem('woven-xray-visible', String(xrayVisible))
  }, [xrayVisible])

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

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <TopBar
        character={characterState}
        provider={providerConfig}
        sessionId={sessionId}
        memoryCount={memories.length}
        onOpenProviderModal={() => setShowProviderModal(true)}
        xrayVisible={xrayVisible}
        onToggleXray={() => setXrayVisible(v => !v)}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Chat panel */}
        <div className={`flex flex-col transition-all duration-300 ${xrayVisible ? 'w-[70%]' : 'w-full'}`}>
          <ChatPanel messages={messages} loading={loading} onSend={handleSend} />
        </div>

        {/* X-Ray panel */}
        {xrayVisible && (
          <div className="w-[30%] transition-all duration-300">
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
    </div>
  )
}
