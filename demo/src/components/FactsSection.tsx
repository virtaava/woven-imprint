import { useState } from 'react'
import { Check, Pencil, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { Fact } from '@/lib/types'

interface Props {
  facts: Fact[]
  onEditObject: (id: string, object: string) => Promise<void>
  onRetract: (id: string) => void
}

export function FactsSection({ facts, onEditObject, onRetract }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)

  const startEdit = (fact: Fact) => {
    setEditingId(fact.id)
    setDraft(fact.object)
  }

  const save = async (fact: Fact) => {
    const value = draft.trim()
    if (!value || value === fact.object) { setEditingId(null); return }
    setBusy(true)
    try {
      await onEditObject(fact.id, value)
      setEditingId(null)
    } finally {
      setBusy(false)
    }
  }

  if (facts.length === 0) {
    return <p className="text-xs text-muted-foreground">No facts recorded yet.</p>
  }

  return (
    <div className="flex flex-col gap-2">
      {facts.map((fact) => {
        const editing = editingId === fact.id
        const since = fact.valid_from?.slice(0, 10)
        return (
          <div key={fact.id} className="flex flex-col gap-1 rounded-md border border-border/50 bg-background/50 p-2">
            {editing ? (
              <div className="flex items-center gap-1">
                <Input
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Escape') setEditingId(null); if (e.key === 'Enter') void save(fact) }}
                  className="h-6 flex-1 text-xs"
                  autoFocus
                />
                <Button variant="ghost" size="icon-xs" title="Save" disabled={busy} onClick={() => void save(fact)}><Check className="size-3" /></Button>
                <Button variant="ghost" size="icon-xs" title="Cancel" onClick={() => setEditingId(null)}><X className="size-3" /></Button>
              </div>
            ) : (
              <div className="flex items-start gap-2">
                <p className="flex-1 text-xs leading-relaxed text-foreground/80">
                  {fact.statement}
                  {since && <span className="ml-1 text-[10px] text-muted-foreground">(since {since})</span>}
                </p>
                <span className="flex shrink-0 items-center gap-1">
                  <Button variant="ghost" size="icon-xs" title="Edit" onClick={() => startEdit(fact)}><Pencil className="size-3" /></Button>
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={() => { if (window.confirm('Retract this fact? It will no longer be considered current.')) onRetract(fact.id) }}
                  >
                    Retract
                  </Button>
                </span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
