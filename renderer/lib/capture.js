import { api } from './api.js'

export const sendCaptureRequest = async (filename, sessionId = null, timestamp = null) => {
  try {
    const res = await fetch('http://localhost:8000/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        filename: filename,
        sessionId: sessionId,
        timestamp: Date.now()
       })
    })

    if (!res.ok) {
      throw new Error(`Capture request failed with status ${res.status}`)
    }

    const data = await res.json()
    
    // Store event if session is active
    if (sessionId) {
      try {
        await api.storeEvent(sessionId, 'ocr', data.message || 'Screenshot captured', {
          filename,
          timestamp: new Date().toISOString(),
          type: 'screenshot',
          source: 'capture'
        })
        console.log('Capture event stored for session:', sessionId)
      } catch (eventError) {
        console.warn('Failed to store capture event:', eventError.message)
        // Don't fail the capture if event storage fails
      }
    }
    return data
  } catch (error) {
    console.error('Capture error:', error)
    throw error
  }
}
