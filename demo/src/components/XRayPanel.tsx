import { useState } from 'react'
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts'
import { Brain, Search, Activity, Zap, Clock, Play, Pencil, Check } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import type { CharacterState, Memory, Relationship, Session } from '@/lib/types'

const TIER_COLORS: Record<string, string> = {
  bedrock: 'text-amber-400 border-amber-400/30 bg-amber-400/10',
  core: 'text-blue-400 border-blue-400/30 bg-blue-400/10',
  buffer: 'text-zinc-400 border-zinc-400/30 bg-zinc-400/10',
}

interface XRayPanelProps {
  character: CharacterState | null
  memories: Memory[]
  relationship: Relationship | null
  onSearchMemory: (query: string) => void
  searchResults: Memory[]
  searchLoading: boolean
  sessions?: Session[]
  activeSessionId?: string | null
  onResumeSession?: (sessionId: string) => void
  onRenameSession?: (sessionId: string, alias: string) => void
}

export function XRayPanel({ character, memories, relationship, onSearchMemory, searchResults, searchLoading, sessions = [], activeSessionId, onResumeSession, onRenameSession }: XRayPanelProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editAlias, setEditAlias] = useState('')

  const radarData = relationship
    ? [
        { dimension: 'Trust', value: relationship.trust },
        { dimension: 'Affection', value: relationship.affection },
        { dimension: 'Respect', value: relationship.respect },
        { dimension: 'Familiarity', value: relationship.familiarity },
        { dimension: 'Tension', value: relationship.tension },
      ]
    : []

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const q = searchQuery.trim()
    if (q) onSearchMemory(q)
  }

  return (
    <ScrollArea className="h-full max-h-[40vh] overflow-auto bg-card/50 lg:max-h-none">
      <div className="flex flex-col gap-4 p-4">
        {/* Header */}
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <Zap className="size-4 text-amber-400" />
          <span>X-Ray</span>
        </div>

        {/* Emotion */}
        {character && (
          <Card size="sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xs">
                <Activity className="size-3.5 text-amber-400" />
                Emotion
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between text-sm">
                <span className="capitalize">{character.emotion.mood}</span>
                <span className="text-xs text-muted-foreground">
                  {Math.round(character.emotion.intensity * 100)}%
                </span>
              </div>
              <div className="mt-1.5 h-1.5 rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-amber-400 transition-all duration-500"
                  style={{ width: `${character.emotion.intensity * 100}%` }}
                />
              </div>
              {character.emotion.cause && (
                <p className="mt-1.5 text-xs text-muted-foreground">{character.emotion.cause}</p>
              )}
              <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                <span>Arc: <span className="capitalize text-foreground">{character.arc.phase}</span></span>
                <span>Tension: {Math.round(character.arc.tension * 100)}%</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Relationship Radar */}
        {relationship && radarData.length > 0 && (
          <Card size="sm">
            <CardHeader>
              <CardTitle className="text-xs">Relationship Radar</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                  <PolarGrid stroke="hsl(0 0% 40% / 0.3)" />
                  <PolarAngleAxis
                    dataKey="dimension"
                    tick={{ fill: 'hsl(0 0% 70%)', fontSize: 10 }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 1]}
                    tick={false}
                    axisLine={false}
                  />
                  <Radar
                    dataKey="value"
                    stroke="#f59e0b"
                    fill="#f59e0b"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        <Separator />

        {/* Memory Feed */}
        <Card size="sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xs">
              <Brain className="size-3.5 text-amber-400" />
              Memory Feed
            </CardTitle>
          </CardHeader>
          <CardContent>
            {memories.length === 0 ? (
              <p className="text-xs text-muted-foreground">No memories yet. Start chatting!</p>
            ) : (
              <div className="flex flex-col gap-2">
                {memories.map((mem, i) => (
                  <div key={i} className="flex flex-col gap-1 rounded-md border border-border/50 bg-background/50 p-2">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className={`text-[10px] px-1.5 py-0 h-4 ${TIER_COLORS[mem.tier] || ''}`}
                      >
                        {mem.tier}
                      </Badge>
                      <span className="text-[10px] text-muted-foreground">
                        {Math.round(mem.importance * 100)}% imp.
                      </span>
                    </div>
                    <p className="text-xs leading-relaxed text-foreground/80">{mem.content}</p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Session History */}
        {sessions.length > 0 && (
          <>
            <Separator />
            <Card size="sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xs">
                  <Clock className="size-3.5 text-amber-400" />
                  Sessions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col gap-1.5">
                  {sessions.slice(0, 10).map((s) => {
                    const isActive = s.id === activeSessionId
                    const isEditing = editingSessionId === s.id
                    const label = s.alias || s.summary?.slice(0, 60) || s.id.slice(0, 12)
                    const date = s.started_at ? new Date(s.started_at + 'Z').toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : ''

                    return (
                      <div
                        key={s.id}
                        className={`flex items-center gap-1.5 rounded-md border p-1.5 text-[11px] ${
                          isActive
                            ? 'border-amber-400/40 bg-amber-400/10'
                            : 'border-border/50 bg-background/50'
                        }`}
                      >
                        {isEditing ? (
                          <form
                            className="flex flex-1 gap-1"
                            onSubmit={(e) => {
                              e.preventDefault()
                              if (editAlias.trim() && onRenameSession) {
                                onRenameSession(s.id, editAlias.trim())
                              }
                              setEditingSessionId(null)
                            }}
                          >
                            <Input
                              value={editAlias}
                              onChange={(e) => setEditAlias(e.target.value)}
                              className="h-5 flex-1 text-[11px]"
                              autoFocus
                            />
                            <Button type="submit" variant="ghost" size="icon" className="size-5">
                              <Check className="size-3" />
                            </Button>
                          </form>
                        ) : (
                          <>
                            <span className="flex-1 truncate">{label}</span>
                            <span className="shrink-0 text-muted-foreground">{date}</span>
                            {onRenameSession && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-5"
                                onClick={() => {
                                  setEditingSessionId(s.id)
                                  setEditAlias(s.alias || '')
                                }}
                              >
                                <Pencil className="size-2.5" />
                              </Button>
                            )}
                            {!isActive && onResumeSession && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-5"
                                onClick={() => onResumeSession(s.id)}
                              >
                                <Play className="size-2.5" />
                              </Button>
                            )}
                            {isActive && (
                              <Badge variant="outline" className="h-4 px-1 text-[9px] text-amber-400 border-amber-400/30">
                                active
                              </Badge>
                            )}
                          </>
                        )}
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </>
        )}

        <Separator />

        {/* Memory Search */}
        <Card size="sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xs">
              <Search className="size-3.5 text-amber-400" />
              Memory Search
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSearch} className="flex gap-2">
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search memories..."
                className="flex-1 text-xs"
              />
              <Button type="submit" variant="outline" size="sm" disabled={searchLoading}>
                <Search className="size-3" />
              </Button>
            </form>
            {searchResults.length > 0 && (
              <div className="mt-3 flex flex-col gap-2">
                {searchResults.map((mem, i) => (
                  <div key={i} className="flex flex-col gap-1 rounded-md border border-border/50 bg-background/50 p-2">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className={`text-[10px] px-1.5 py-0 h-4 ${TIER_COLORS[mem.tier] || ''}`}
                      >
                        {mem.tier}
                      </Badge>
                    </div>
                    <p className="text-xs leading-relaxed text-foreground/80">{mem.content}</p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  )
}
