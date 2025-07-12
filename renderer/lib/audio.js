export async function sendListenRequest(filename) {
  const res = await fetch('http://localhost:8000/asr', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ filename }),
  })

  if (!res.ok) {
    throw new Error(`ASR request failed with status ${res.status}`)
  }

  const data = await res.json()
  console.log('Backend responded with:', data)
  return data.message
}

