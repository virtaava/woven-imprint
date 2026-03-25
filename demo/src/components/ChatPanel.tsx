import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { BookOpen, Send, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { ChatMessage } from '@/lib/types'

const SUGGESTED_PROMPTS = [
  'What exactly is woven-imprint?',
  'Show me how your memory works',
  'How would I add this to my own app?',
]

interface ChatPanelProps {
  messages: ChatMessage[]
  loading: boolean
  onSend: (text: string) => void
}

export function ChatPanel({ messages, loading, onSend }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const showChips = messages.length <= 1 // Only show on first greeting

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    onSend(text)
  }

  const handleChipClick = (prompt: string) => {
    if (loading) return
    onSend(prompt)
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <ScrollArea className="flex-1 overflow-auto">
        <div className="flex flex-col gap-4 p-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            >
              {/* Avatar */}
              {msg.role === 'assistant' && (
                <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-amber-400/10 text-amber-400">
                  <BookOpen className="size-4" />
                </div>
              )}

              {/* Bubble */}
              <div
                className={`max-w-[75%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-foreground'
                }`}
              >
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    code: ({ children, className }) => {
                      const isBlock = className?.includes('language-')
                      return isBlock ? (
                        <pre className="my-2 overflow-x-auto rounded-md bg-black/20 p-3 text-xs">
                          <code>{children}</code>
                        </pre>
                      ) : (
                        <code className="rounded bg-black/20 px-1 py-0.5 text-xs">{children}</code>
                      )
                    },
                    ul: ({ children }) => <ul className="mb-2 ml-4 list-disc">{children}</ul>,
                    ol: ({ children }) => <ol className="mb-2 ml-4 list-decimal">{children}</ol>,
                    li: ({ children }) => <li className="mb-0.5">{children}</li>,
                    strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                    a: ({ children, href }) => (
                      <a href={href} target="_blank" rel="noopener noreferrer" className="text-amber-400 underline underline-offset-2 hover:text-amber-300">
                        {children}
                      </a>
                    ),
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="flex gap-3">
              <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-amber-400/10 text-amber-400">
                <BookOpen className="size-4" />
              </div>
              <div className="flex items-center gap-2 rounded-xl bg-muted px-4 py-2.5 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                <span>Meridian is thinking...</span>
              </div>
            </div>
          )}

          {/* Suggested prompt chips */}
          {showChips && !loading && (
            <div className="flex flex-wrap gap-2 pt-2">
              {SUGGESTED_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => handleChipClick(prompt)}
                  className="rounded-full border border-amber-400/30 bg-amber-400/5 px-3 py-1.5 text-xs text-amber-400 transition-colors hover:bg-amber-400/15"
                >
                  {prompt}
                </button>
              ))}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="flex gap-2 border-t border-border bg-card p-3"
      >
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message Meridian..."
          disabled={loading}
          className="flex-1"
        />
        <Button type="submit" size="default" disabled={!input.trim() || loading}>
          <Send className="size-4" />
        </Button>
      </form>
    </div>
  )
}
