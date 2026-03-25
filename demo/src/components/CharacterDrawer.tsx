import { useState, useRef } from 'react'
import { Users, UserPlus, Trash2, Download, Upload, FileText, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  createCharacter,
  deleteCharacter,
  exportCharacter,
  importCharacter,
  migrateCharacter,
} from '@/lib/api'
import type { CharacterSummary } from '@/lib/types'

interface CharacterDrawerProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  characters: CharacterSummary[]
  activeCharacterId: string | null
  onSelectCharacter: (id: string) => void
  onRefreshCharacters: () => Promise<void>
}

export function CharacterDrawer({
  open,
  onOpenChange,
  characters,
  activeCharacterId,
  onSelectCharacter,
  onRefreshCharacters,
}: CharacterDrawerProps) {
  // Create form
  const [createName, setCreateName] = useState('')
  const [createPersonality, setCreatePersonality] = useState('')
  const [createBackstory, setCreateBackstory] = useState('')
  const [createStyle, setCreateStyle] = useState('')
  const [createBirthdate, setCreateBirthdate] = useState('')
  const [creating, setCreating] = useState(false)

  // Migrate form
  const [migrateName, setMigrateName] = useState('')
  const [migrateText, setMigrateText] = useState('')
  const [migrating, setMigrating] = useState(false)

  // Status feedback
  const [status, setStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [exporting, setExporting] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const showStatus = (type: 'success' | 'error', message: string) => {
    setStatus({ type, message })
    setTimeout(() => setStatus(null), 3000)
  }

  const handleCreate = async () => {
    if (!createName.trim()) return
    setCreating(true)
    try {
      const result = await createCharacter({
        name: createName.trim(),
        personality: createPersonality.trim() || undefined,
        backstory: createBackstory.trim() || undefined,
        speaking_style: createStyle.trim() || undefined,
        birthdate: createBirthdate.trim() || undefined,
      })
      await onRefreshCharacters()
      const newId = result.id || result.character_id
      if (newId) onSelectCharacter(newId)
      setCreateName('')
      setCreatePersonality('')
      setCreateBackstory('')
      setCreateStyle('')
      setCreateBirthdate('')
      showStatus('success', `Created "${createName.trim()}"`)
    } catch (err: any) {
      showStatus('error', err.message || 'Failed to create character')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteCharacter(id)
      await onRefreshCharacters()
      setConfirmDelete(null)
      showStatus('success', 'Character deleted')
    } catch (err: any) {
      showStatus('error', err.message || 'Failed to delete character')
    }
  }

  const handleExport = async (id: string, name: string) => {
    setExporting(id)
    try {
      const data = await exportCharacter(id)
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${name}.json`
      a.click()
      URL.revokeObjectURL(url)
      showStatus('success', `Exported "${name}"`)
    } catch (err: any) {
      showStatus('error', err.message || 'Failed to export character')
    } finally {
      setExporting(null)
    }
  }

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const result = await importCharacter(data)
      await onRefreshCharacters()
      const newId = result.id || result.character_id
      if (newId) onSelectCharacter(newId)
      showStatus('success', `Imported "${result.name || file.name}"`)
    } catch (err: any) {
      showStatus('error', err.message || 'Failed to import character')
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleMigrate = async () => {
    if (!migrateName.trim() || !migrateText.trim()) return
    setMigrating(true)
    try {
      const result = await migrateCharacter(migrateName.trim(), migrateText.trim())
      await onRefreshCharacters()
      const newId = result.id || result.character_id
      if (newId) onSelectCharacter(newId)
      setMigrateName('')
      setMigrateText('')
      showStatus('success', `Migrated "${migrateName.trim()}"`)
    } catch (err: any) {
      showStatus('error', err.message || 'Failed to migrate character')
    } finally {
      setMigrating(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Users className="size-4 text-amber-400" />
            Character Management
          </DialogTitle>
          <DialogDescription>
            Select, create, import, or migrate characters.
          </DialogDescription>
        </DialogHeader>

        {/* Status toast */}
        {status && (
          <div
            className={`flex items-center gap-2 rounded-md border p-2 text-xs ${
              status.type === 'success'
                ? 'border-emerald-400/30 bg-emerald-400/5 text-emerald-400'
                : 'border-red-400/30 bg-red-400/5 text-red-400'
            }`}
          >
            {status.type === 'success' ? <CheckCircle className="size-3.5" /> : <AlertCircle className="size-3.5" />}
            {status.message}
          </div>
        )}

        <Tabs defaultValue="characters" className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="w-full">
            <TabsTrigger value="characters" className="flex-1">Characters</TabsTrigger>
            <TabsTrigger value="create" className="flex-1">Create</TabsTrigger>
            <TabsTrigger value="import" className="flex-1">Import</TabsTrigger>
          </TabsList>

          {/* Characters list */}
          <TabsContent value="characters" className="flex-1 overflow-hidden mt-2">
            <ScrollArea className="h-[45vh]">
              <div className="flex flex-col gap-1 pr-3">
                {characters.length === 0 ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    No characters yet. Create one to get started.
                  </p>
                ) : (
                  characters.map((char) => {
                    const id = char.id || char.character_id || ''
                    const isActive = id === activeCharacterId
                    return (
                      <div
                        key={id}
                        className={`group flex items-center justify-between rounded-md px-3 py-2 transition-colors cursor-pointer ${
                          isActive
                            ? 'bg-amber-400/10 border border-amber-400/30 text-amber-400'
                            : 'hover:bg-accent border border-transparent'
                        }`}
                        onClick={() => {
                          if (!isActive) {
                            onSelectCharacter(id)
                            onOpenChange(false)
                          }
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">{char.name}</span>
                          {isActive && (
                            <span className="text-[10px] uppercase tracking-wider text-amber-400/70">active</span>
                          )}
                        </div>
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleExport(id, char.name)
                            }}
                            disabled={exporting === id}
                            title="Export character"
                          >
                            {exporting === id ? (
                              <Loader2 className="size-3.5 animate-spin" />
                            ) : (
                              <Download className="size-3.5" />
                            )}
                          </Button>
                          {confirmDelete === id ? (
                            <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 px-2 text-xs text-red-400 hover:text-red-300"
                                onClick={() => handleDelete(id)}
                              >
                                Confirm
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 px-2 text-xs"
                                onClick={() => setConfirmDelete(null)}
                              >
                                Cancel
                              </Button>
                            </div>
                          ) : (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-7 text-muted-foreground hover:text-red-400"
                              onClick={(e) => {
                                e.stopPropagation()
                                setConfirmDelete(id)
                              }}
                              title="Delete character"
                            >
                              <Trash2 className="size-3.5" />
                            </Button>
                          )}
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Create tab */}
          <TabsContent value="create" className="flex-1 overflow-hidden mt-2">
            <ScrollArea className="h-[45vh]">
              <div className="flex flex-col gap-3 pr-3">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Name *</label>
                  <Input
                    value={createName}
                    onChange={(e) => setCreateName(e.target.value)}
                    placeholder="Character name"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Personality</label>
                  <Input
                    value={createPersonality}
                    onChange={(e) => setCreatePersonality(e.target.value)}
                    placeholder="e.g. Warm, curious, slightly mischievous"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Backstory</label>
                  <textarea
                    className="flex min-h-[60px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    value={createBackstory}
                    onChange={(e) => setCreateBackstory(e.target.value)}
                    placeholder="Character's background and history..."
                    rows={3}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Speaking Style</label>
                  <Input
                    value={createStyle}
                    onChange={(e) => setCreateStyle(e.target.value)}
                    placeholder="e.g. Formal, uses metaphors"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Birthdate</label>
                  <Input
                    value={createBirthdate}
                    onChange={(e) => setCreateBirthdate(e.target.value)}
                    placeholder="e.g. 1995-03-15"
                  />
                </div>
                <Button onClick={handleCreate} disabled={creating || !createName.trim()}>
                  {creating ? <Loader2 className="size-3.5 animate-spin" /> : <UserPlus className="size-3.5" />}
                  Create Character
                </Button>
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Import / Migrate tab */}
          <TabsContent value="import" className="flex-1 overflow-hidden mt-2">
            <ScrollArea className="h-[45vh]">
              <div className="flex flex-col gap-4 pr-3">
                {/* JSON Import */}
                <div className="flex flex-col gap-2">
                  <h3 className="flex items-center gap-2 text-sm font-medium">
                    <Upload className="size-3.5 text-amber-400" />
                    Import from JSON
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    Upload a previously exported character JSON file.
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={handleImportFile}
                  />
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="size-3.5" />
                    Choose JSON File
                  </Button>
                </div>

                <Separator />

                {/* Migrate */}
                <div className="flex flex-col gap-2">
                  <h3 className="flex items-center gap-2 text-sm font-medium">
                    <FileText className="size-3.5 text-amber-400" />
                    Migrate from Text
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    Paste Custom GPT instructions, persona text, or a character card to create a character automatically.
                  </p>
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Name *</label>
                    <Input
                      value={migrateName}
                      onChange={(e) => setMigrateName(e.target.value)}
                      placeholder="Character name"
                    />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Source Text *</label>
                    <textarea
                      className="flex min-h-[120px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      value={migrateText}
                      onChange={(e) => setMigrateText(e.target.value)}
                      placeholder="Paste your character instructions, persona, or character card text here..."
                      rows={6}
                    />
                  </div>
                  <Button
                    onClick={handleMigrate}
                    disabled={migrating || !migrateName.trim() || !migrateText.trim()}
                  >
                    {migrating ? <Loader2 className="size-3.5 animate-spin" /> : <FileText className="size-3.5" />}
                    Migrate Character
                  </Button>
                </div>
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
