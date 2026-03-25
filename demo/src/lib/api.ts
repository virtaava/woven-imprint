const getToken = () => (window as any).__WOVEN_TOKEN__ || ''

const headers = () => ({
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${getToken()}`,
})

const API_BASE = ''  // Same origin

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`)
  return res.json()
}

export async function fetchCharacters() {
  const res = await fetch(`${API_BASE}/api/characters`, { headers: headers() })
  return res.json()
}

export async function fetchCharacterState(id: string) {
  const res = await fetch(`${API_BASE}/api/characters/${id}`, { headers: headers() })
  return res.json()
}

export async function startSession(id: string) {
  const res = await fetch(`${API_BASE}/api/characters/${id}/session`, { method: 'POST', headers: headers() })
  return res.json()
}

export async function sendMessage(model: string, messages: { role: string; content: string }[]) {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ model, messages }),
  })
  return res.json()
}

export async function recallMemories(characterId: string, query: string, limit = 5) {
  const params = new URLSearchParams({ character_id: characterId, query, limit: String(limit) })
  const res = await fetch(`${API_BASE}/api/memory?${params}`, { headers: headers() })
  return res.json()
}

export async function fetchRelationship(charId: string, targetId: string) {
  const res = await fetch(`${API_BASE}/api/relationships/${charId}/${targetId}`, { headers: headers() })
  return res.json()
}

export async function getProviderConfig() {
  const res = await fetch(`${API_BASE}/api/config/provider`, { headers: headers() })
  return res.json()
}

export async function updateProviderConfig(config: { provider: string; model: string; api_key?: string; base_url?: string }) {
  const res = await fetch(`${API_BASE}/api/config/provider`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify(config),
  })
  return res.json()
}

export async function testProviderConnection(config: { provider: string; model: string; api_key?: string; base_url?: string }) {
  const res = await fetch(`${API_BASE}/api/config/provider/test`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify(config),
  })
  return res.json()
}
