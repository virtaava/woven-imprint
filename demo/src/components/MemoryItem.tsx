import { useState } from 'react'
import { Pin, PinOff, Pencil, Trash2, Check, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { Memory } from '@/lib/types'

const TIER_COLORS: Record<string, string> = {
  bedrock: 'text-amber-400 border-amber-400/30 bg-amber-400/10',
  core: 'text-blue-400 border-blue-400/30 bg-blue-400/10',
  buffer: 'text-zinc-400 border-zinc-400/30 bg-zinc-400/10',
}

interface Props {
  memory: Memory
  onPin: (id: string, pinned: boolean) => void
  onEdit: (id: string, content: string) => Promise<void>
  onDelete: (id: string) => void
}

export function MemoryItem({ memory, onPin, onEdit, onDelete }: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(memory.content)
  const [busy, setBusy] = useState(false)
  const pinned = memory.metadata?.pinned === true
  const date = memory.created_at?.slice(0, 10)

  const save = async () => {
    if (!draft.trim() || draft === memory.content) { setEditing(false); return }
    setBusy(true)
    try { await onEdit(memory.id, draft.trim()); setEditing(false) } finally { setBusy(false) }
  }

  return (
    <div className={`flex flex-col gap-1 rounded-md border p-2 ${pinned ? 'border-amber-400/40 bg-amber-400/5' : 'border-border/50 bg-background/50'}`}>
      <div className="flex items-center gap-2">
        <Badge variant="outline" className={`text-[10px] px-1.5 py-0 h-4 ${TIER_COLORS[memory.tier] || ''}`}>{memory.tier}</Badge>
        <span className="text-[10px] text-muted-foreground">{Math.round(memory.importance * 100)}% imp.</span>
        {date && <span className="text-[10px] text-muted-foreground">{date}</span>}
        <span className="ml-auto flex items-center gap-0.5">
          <Button variant="ghost" size="icon-xs" title={pinned ? 'Unpin' : 'Pin (always in prompt)'} onClick={() => onPin(memory.id, !pinned)}>
            {pinned ? <PinOff className="size-3" /> : <Pin className="size-3" />}
          </Button>
          <Button variant="ghost" size="icon-xs" title="Edit" onClick={() => { setDraft(memory.content); setEditing(true) }}><Pencil className="size-3" /></Button>
          <Button variant="ghost" size="icon-xs" title="Delete" onClick={() => { if (window.confirm('Forget this memory? This cannot be undone.')) onDelete(memory.id) }}><Trash2 className="size-3" /></Button>
        </span>
      </div>
      {editing ? (
        <div className="flex flex-col gap-1">
          <textarea className="w-full rounded-md border border-border bg-background p-1 text-xs" rows={3} value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') setEditing(false); if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void save() }} />
          <div className="flex gap-1">
            <Button size="xs" disabled={busy} onClick={() => void save()}><Check className="size-3" /> Save</Button>
            <Button size="xs" variant="ghost" onClick={() => setEditing(false)}><X className="size-3" /> Cancel</Button>
          </div>
        </div>
      ) : (
        <p className="text-xs leading-relaxed text-foreground/80">{memory.content}</p>
      )}
    </div>
  )
}
