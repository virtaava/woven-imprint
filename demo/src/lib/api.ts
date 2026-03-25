const headers = () => ({
  'Content-Type': 'application/json',
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

export async function fetchAvailableModels(provider: string, base_url?: string, api_key?: string) {
  const params = new URLSearchParams({ provider })
  if (base_url) params.set('base_url', base_url)
  if (api_key) params.set('api_key', api_key)
  const res = await fetch(`${API_BASE}/api/config/models?${params}`, { headers: headers() })
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

export async function createCharacter(data: { name: string; personality?: string; backstory?: string; speaking_style?: string; birthdate?: string }) {
  const res = await fetch(`${API_BASE}/api/characters`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify(data),
  })
  return res.json()
}

export async function deleteCharacter(id: string) {
  const res = await fetch(`${API_BASE}/api/characters/${id}`, {
    method: 'DELETE',
    headers: headers(),
  })
  return res.json()
}

export async function exportCharacter(id: string) {
  const res = await fetch(`${API_BASE}/api/characters/${id}/export`, { headers: headers() })
  return res.json()
}

export async function importCharacter(data: object) {
  const res = await fetch(`${API_BASE}/api/characters/import`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify(data),
  })
  return res.json()
}

export async function importCharacterFile(file: File, name?: string) {
  const formData = new FormData()
  formData.append('file', file)
  if (name) formData.append('name', name)
  const res = await fetch(`${API_BASE}/api/characters/import-file`, {
    method: 'POST',
    // No Content-Type — browser sets multipart boundary
    body: formData,
  })
  return res.json()
}

export async function migrateCharacter(name: string, text: string) {
  const res = await fetch(`${API_BASE}/api/characters/migrate`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ name, text }),
  })
  return res.json()
}

export async function reflectCharacter(id: string) {
  const res = await fetch(`${API_BASE}/api/characters/${id}/reflect`, {
    method: 'POST',
    headers: headers(),
  })
  return res.json()
}

export async function endSession(characterId: string) {
  const res = await fetch(`${API_BASE}/api/characters/${characterId}/session`, {
    method: 'DELETE',
    headers: headers(),
  })
  return res.json()
}

export async function fetchSessions(characterId: string, limit = 20) {
  const params = new URLSearchParams({ limit: String(limit) })
  const res = await fetch(`${API_BASE}/api/characters/${characterId}/sessions?${params}`, { headers: headers() })
  return res.json()
}

export async function renameSession(characterId: string, sessionId: string, alias: string) {
  const res = await fetch(`${API_BASE}/api/characters/${characterId}/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: headers(),
    body: JSON.stringify({ alias }),
  })
  return res.json()
}

export async function resumeSession(characterId: string, sessionId: string) {
  const res = await fetch(`${API_BASE}/api/characters/${characterId}/sessions/${sessionId}/resume`, {
    method: 'POST',
    headers: headers(),
  })
  return res.json()
}
