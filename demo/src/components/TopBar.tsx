import { BookOpen, Settings, Wifi, WifiOff, PanelRightOpen, PanelRightClose, Users, ChevronDown } from 'lucide-react'
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
  onOpenCharacterDrawer: () => void
  xrayVisible: boolean
  onToggleXray: () => void
}

export function TopBar({ character, provider, sessionId, memoryCount, onOpenProviderModal, onOpenCharacterDrawer, xrayVisible, onToggleXray }: TopBarProps) {
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

      {/* Center: character selector + info */}
      <div className="flex items-center gap-4">
        <button
          onClick={onOpenCharacterDrawer}
          className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:border-amber-400/30 hover:bg-amber-400/5"
          title="Manage characters"
        >
          <Users className="size-3.5 text-amber-400" />
          <span className="font-medium">{character?.name || 'No character'}</span>
          <ChevronDown className="size-3 text-muted-foreground" />
        </button>
        {sessionId && (
          <Badge variant="outline" className="text-xs text-amber-400 border-amber-400/30">
            Session active
          </Badge>
        )}
        {character && (
          <span className="text-xs text-muted-foreground">
            {memoryCount} memories
          </span>
        )}
      </div>

      {/* Right: provider + settings */}
      <div className="flex items-center gap-3">
        {provider && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {(provider.api_key_configured || provider.provider === 'ollama') ? (
              <Wifi className="size-3.5 text-emerald-400" />
            ) : (
              <WifiOff className="size-3.5 text-destructive" />
            )}
            <span className="capitalize">{provider.provider}</span>
            <span className="text-foreground">{provider.model}</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleXray}
          title={xrayVisible ? 'Hide X-Ray panel' : 'Show X-Ray panel'}
        >
          {xrayVisible ? <PanelRightClose className="size-4" /> : <PanelRightOpen className="size-4" />}
        </Button>
        <Button variant="ghost" size="icon" onClick={onOpenProviderModal}>
          <Settings className="size-4" />
        </Button>
      </div>
    </div>
  )
}
