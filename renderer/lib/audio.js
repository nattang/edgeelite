export async function sendListenRequest() {
  const res = await fetch('http://localhost:8000/asr', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ filename: 'dummy_audio.wav' }),  // placeholder
  })

  const data = await res.json()
  console.log('Backend responded with:', data)
  return data.message
}
