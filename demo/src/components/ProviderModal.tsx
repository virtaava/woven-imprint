import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Loader2, RefreshCw } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { testProviderConnection, updateProviderConfig, fetchAvailableModels } from '@/lib/api'
import type { ProviderConfig } from '@/lib/types'

// provider id maps to woven-imprint's provider system.
// DeepSeek, NVIDIA, and Custom all use 'openai' provider with custom base_url.
const PROVIDERS = [
  { id: 'ollama', label: 'Ollama', needsKey: false, defaultModel: 'llama3.2', defaultUrl: 'http://localhost:11434', showUrl: true, keyHint: '' },
  { id: 'openai', label: 'OpenAI', needsKey: true, defaultModel: 'gpt-4o-mini', showUrl: false, keyHint: 'sk-...' },
  { id: 'anthropic', label: 'Anthropic', needsKey: true, defaultModel: 'claude-sonnet-4-5-20250514', showUrl: false, keyHint: 'sk-ant-...' },
  { id: 'openai', label: 'DeepSeek', needsKey: true, defaultModel: 'deepseek-chat', defaultUrl: 'https://api.deepseek.com/v1', showUrl: false, preset: 'deepseek', keyHint: 'sk-...' },
  { id: 'openai', label: 'NVIDIA NIM', needsKey: true, defaultModel: 'meta/llama-3.1-8b-instruct', defaultUrl: 'https://integrate.api.nvidia.com/v1', showUrl: false, preset: 'nvidia', keyHint: 'nvapi-...' },
  { id: 'openai', label: 'Custom', needsKey: true, defaultModel: '', showUrl: true, preset: 'custom', keyHint: 'API key (if required)', description: 'Any OpenAI-compatible API' },
]

interface ProviderModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  currentConfig: ProviderConfig | null
  onSaved: (config: ProviderConfig) => void
}

export function ProviderModal({ open, onOpenChange, currentConfig, onSaved }: ProviderModalProps) {
  // selectedKey uniquely identifies the preset (preset field or label)
  const [selectedKey, setSelectedKey] = useState('OpenAI')
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [testPassed, setTestPassed] = useState(false)
  const [saving, setSaving] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)

  const getKey = (p: typeof PROVIDERS[number]) => p.preset || p.label
  const selected = PROVIDERS.find((p) => getKey(p) === selectedKey)

  useEffect(() => {
    if (currentConfig) {
      // Find the best matching preset for the current config
      const match = PROVIDERS.find(
        (p) => p.id === currentConfig.provider && p.defaultUrl === currentConfig.base_url
      ) || PROVIDERS.find((p) => p.id === currentConfig.provider)
      if (match) setSelectedKey(getKey(match))
      setModel(currentConfig.model || '')
      setBaseUrl(currentConfig.base_url || '')
    }
  }, [currentConfig])

  // When preset changes, update provider/model/url
  useEffect(() => {
    if (!selected) return
    setProvider(selected.id)
    setModel(selected.defaultModel)
    setBaseUrl(selected.defaultUrl || '')
    setTestResult(null)
    setTestPassed(false)
    setAvailableModels([])

    if (open) {
      loadModels(selected.id, selected.defaultUrl || '')
    }
  }, [selectedKey])

  // Re-fetch when modal opens
  useEffect(() => {
    if (open) {
      loadModels(provider, baseUrl)
    }
  }, [open])

  const loadModels = async (prov: string, url?: string) => {
    setLoadingModels(true)
    try {
      const res = await fetchAvailableModels(prov, url || undefined)
      setAvailableModels(res.models || [])
    } catch {
      setAvailableModels([])
    } finally {
      setLoadingModels(false)
    }
  }

  const handleRefreshModels = () => {
    loadModels(provider, baseUrl)
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await testProviderConnection({
        provider,
        model,
        api_key: apiKey || undefined,
        base_url: baseUrl || undefined,
      })
      const ok = res.ok ?? res.success ?? true
      setTestResult({ ok, message: res.message || 'Connection successful' })
      setTestPassed(ok)
    } catch (err: any) {
      setTestResult({ ok: false, message: err.message || 'Connection failed' })
      setTestPassed(false)
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const res = await updateProviderConfig({
        provider,
        model,
        api_key: apiKey || undefined,
        base_url: baseUrl || undefined,
      })
      onSaved(res)
      onOpenChange(false)
    } catch {
      // keep modal open on error
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Provider Configuration</DialogTitle>
          <DialogDescription>
            Configure the LLM provider for Meridian's responses.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          {/* Provider selector */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-muted-foreground">Provider</label>
            <div className="flex flex-wrap gap-2">
              {PROVIDERS.map((p) => {
                const key = p.preset || p.label
                return (
                  <button
                    key={key}
                    onClick={() => setSelectedKey(key)}
                    className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                      selectedKey === key
                        ? 'border-amber-400 bg-amber-400/10 text-amber-400'
                        : 'border-border text-muted-foreground hover:text-foreground'
                    }`}
                    title={(p as any).description || p.label}
                  >
                    {p.label}
                  </button>
                )
              })}
            </div>
            {selected && (selected as any).description && (
              <p className="text-xs text-muted-foreground">{(selected as any).description}</p>
            )}
          </div>

          {/* API Key */}
          {selected?.needsKey && (
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">API Key</label>
              <Input
                type="password"
                value={apiKey}
                onChange={(e) => { setApiKey(e.target.value); setTestPassed(false); setTestResult(null) }}
                placeholder={currentConfig?.api_key_configured ? 'Key configured (leave empty to keep)' : (selected?.keyHint || 'API key')}
              />
            </div>
          )}

          {/* Base URL */}
          {selected?.showUrl && (
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                Base URL {selected.id !== 'ollama' && '(optional)'}
              </label>
              <div className="flex gap-2">
                <Input
                  value={baseUrl}
                  onChange={(e) => { setBaseUrl(e.target.value); setTestPassed(false); setTestResult(null) }}
                  placeholder={provider === 'ollama' ? 'http://localhost:11434' : 'https://api.openai.com/v1'}
                  className="flex-1"
                />
                {selected?.id === 'ollama' && (
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleRefreshModels}
                    disabled={loadingModels}
                    title="Refresh models from this URL"
                  >
                    <RefreshCw className={`size-3.5 ${loadingModels ? 'animate-spin' : ''}`} />
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Model selector */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">Model</label>
              {loadingModels && (
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Loader2 className="size-3 animate-spin" /> Loading models...
                </span>
              )}
            </div>

            {availableModels.length > 0 ? (
              <ScrollArea className="max-h-40 rounded-md border">
                <div className="flex flex-col p-1">
                  {availableModels.map((m) => (
                    <button
                      key={m}
                      onClick={() => { setModel(m); setTestPassed(false); setTestResult(null) }}
                      className={`rounded px-2 py-1.5 text-left text-xs transition-colors ${
                        model === m
                          ? 'bg-amber-400/10 text-amber-400'
                          : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <Input
                value={model}
                onChange={(e) => { setModel(e.target.value); setTestPassed(false); setTestResult(null) }}
                placeholder="Model name"
              />
            )}

            {/* Always show the manual input if models are loaded, for custom entries */}
            {availableModels.length > 0 && (
              <Input
                value={model}
                onChange={(e) => { setModel(e.target.value); setTestPassed(false); setTestResult(null) }}
                placeholder="Or type a model name..."
                className="text-xs"
              />
            )}
          </div>

          {/* Test result */}
          {testResult && (
            <div
              className={`flex items-center gap-2 rounded-md border p-2 text-xs ${
                testResult.ok
                  ? 'border-emerald-400/30 bg-emerald-400/5 text-emerald-400'
                  : 'border-red-400/30 bg-red-400/5 text-red-400'
              }`}
            >
              {testResult.ok ? <CheckCircle className="size-3.5" /> : <XCircle className="size-3.5" />}
              {testResult.message}
            </div>
          )}
        </div>

        <DialogFooter className="flex-col items-stretch gap-2 sm:flex-col">
          {!testPassed && model && (
            <p className="text-xs text-muted-foreground text-center">
              Test the connection before saving to verify the model is reachable.
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={handleTest} disabled={testing || !model}>
              {testing && <Loader2 className="size-3.5 animate-spin" />}
              Test Connection
            </Button>
            <Button onClick={handleSave} disabled={saving || !model || !testPassed}>
              {saving && <Loader2 className="size-3.5 animate-spin" />}
              Save
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
