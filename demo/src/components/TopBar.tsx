import { BookOpen, Settings, Wifi, WifiOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import type { CharacterState, ProviderConfig } from '@/lib/types'

interface TopBarProps {
  character: CharacterState | null
  provider: ProviderConfig | null
  sessionId: string | null
  memoryCount: number
  onOpenProviderModal: () => void
}

export function TopBar({ character, provider, sessionId, memoryCount, onOpenProviderModal }: TopBarProps) {
  return (
    <div className="flex h-14 items-center justify-between border-b border-border bg-card px-4">
      {/* Left: branding */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-amber-400">
          <BookOpen className="size-5" />
          <span className="text-base font-semibold tracking-tight">woven-imprint</span>
        </div>
        <Separator orientation="vertical" className="h-6" />
        <span className="text-sm text-muted-foreground">Interactive Demo</span>
      </div>

      {/* Center: character info */}
      <div className="flex items-center gap-4">
        {character && (
          <>
            <span className="text-sm font-medium">{character.name}</span>
            {sessionId && (
              <Badge variant="outline" className="text-xs text-amber-400 border-amber-400/30">
                Session active
              </Badge>
            )}
            <span className="text-xs text-muted-foreground">
              {memoryCount} memories
            </span>
          </>
        )}
      </div>

      {/* Right: provider + settings */}
      <div className="flex items-center gap-3">
        {provider && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {provider.api_key_configured ? (
              <Wifi className="size-3.5 text-emerald-400" />
            ) : (
              <WifiOff className="size-3.5 text-destructive" />
            )}
            <span>{provider.provider}/{provider.model}</span>
          </div>
        )}
        <Button variant="ghost" size="icon-sm" onClick={onOpenProviderModal}>
          <Settings className="size-4" />
        </Button>
      </div>
    </div>
  )
}
