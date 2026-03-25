import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Loader2 } from 'lucide-react'
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
import { testProviderConnection, updateProviderConfig } from '@/lib/api'
import type { ProviderConfig } from '@/lib/types'

const PROVIDERS = [
  { id: 'openai', label: 'OpenAI', needsKey: true, defaultModel: 'gpt-4o-mini' },
  { id: 'anthropic', label: 'Anthropic', needsKey: true, defaultModel: 'claude-sonnet-4-20250514' },
  { id: 'ollama', label: 'Ollama', needsKey: false, defaultModel: 'llama3.2', defaultUrl: 'http://localhost:11434' },
  { id: 'deepseek', label: 'DeepSeek', needsKey: true, defaultModel: 'deepseek-chat' },
]

interface ProviderModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  currentConfig: ProviderConfig | null
  onSaved: (config: ProviderConfig) => void
}

export function ProviderModal({ open, onOpenChange, currentConfig, onSaved }: ProviderModalProps) {
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [saving, setSaving] = useState(false)

  const providerInfo = PROVIDERS.find((p) => p.id === provider)

  useEffect(() => {
    if (currentConfig) {
      setProvider(currentConfig.provider || 'openai')
      setModel(currentConfig.model || '')
      setBaseUrl(currentConfig.base_url || '')
    }
  }, [currentConfig])

  useEffect(() => {
    const info = PROVIDERS.find((p) => p.id === provider)
    if (info) {
      if (!model || PROVIDERS.some((p) => p.defaultModel === model)) {
        setModel(info.defaultModel)
      }
      if (info.defaultUrl && !baseUrl) {
        setBaseUrl(info.defaultUrl)
      } else if (!info.defaultUrl && PROVIDERS.some((p) => p.defaultUrl === baseUrl)) {
        setBaseUrl('')
      }
    }
    setTestResult(null)
  }, [provider])

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
      setTestResult({ ok: res.ok ?? res.success ?? true, message: res.message || 'Connection successful' })
    } catch (err: any) {
      setTestResult({ ok: false, message: err.message || 'Connection failed' })
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
              {PROVIDERS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setProvider(p.id)}
                  className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                    provider === p.id
                      ? 'border-amber-400 bg-amber-400/10 text-amber-400'
                      : 'border-border text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* API Key */}
          {providerInfo?.needsKey && (
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">API Key</label>
              <Input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={currentConfig?.api_key_configured ? 'Key configured (leave empty to keep)' : 'sk-...'}
              />
            </div>
          )}

          {/* Model */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-muted-foreground">Model</label>
            <Input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="Model name"
            />
          </div>

          {/* Base URL */}
          {(provider === 'ollama' || provider === 'openai') && (
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                Base URL {provider !== 'ollama' && '(optional)'}
              </label>
              <Input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={provider === 'ollama' ? 'http://localhost:11434' : 'https://api.openai.com/v1'}
              />
            </div>
          )}

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

        <DialogFooter>
          <Button variant="outline" onClick={handleTest} disabled={testing || !model}>
            {testing && <Loader2 className="size-3.5 animate-spin" />}
            Test Connection
          </Button>
          <Button onClick={handleSave} disabled={saving || !model}>
            {saving && <Loader2 className="size-3.5 animate-spin" />}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
