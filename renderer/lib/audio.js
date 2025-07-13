// Web Audio API-based recording system
import { api } from './api.js'

let mediaRecorder = null
let audioChunks = []
let isRecording = false

export async function startRecording() {
  try {
    console.log('🎤 Starting audio recording...')
    
    // Request microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      } 
    })
    
    console.log('🎤 Microphone access granted')
    
    // Create MediaRecorder
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    })
    
    audioChunks = []
    isRecording = true
    
    console.log('🎤 MediaRecorder created, isRecording set to:', isRecording)
    
    // Collect audio data
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data)
        console.log('🎤 Audio chunk received, size:', event.data.size)
      }
    }
    
    // Start recording
    mediaRecorder.start(100) // Collect data every 100ms
    
    console.log('🎤 Audio recording started successfully')
    return true
    
  } catch (error) {
    console.error('❌ Failed to start recording:', error)
    isRecording = false
    throw error
  }
}

export async function stopRecording() {
  return new Promise((resolve, reject) => {
    console.log('🎤 Stopping recording, isRecording:', isRecording, 'mediaRecorder:', !!mediaRecorder)
    
    if (!mediaRecorder || !isRecording) {
      console.error('❌ No active recording to stop')
      reject(new Error('No active recording'))
      return
    }
    
    mediaRecorder.onstop = async () => {
      try {
        console.log('🎤 MediaRecorder stopped, processing audio...')
        
        // Create audio blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
        console.log('🎤 Audio blob created, size:', audioBlob.size)
        
        // Convert to WAV format (simplified)
        const arrayBuffer = await audioBlob.arrayBuffer()
        const wavBlob = await convertToWav(arrayBuffer)
        console.log('🎤 WAV blob created, size:', wavBlob.size)
        
        // Convert WAV blob to buffer for Electron
        const wavArrayBuffer = await wavBlob.arrayBuffer()
        const wavBuffer = new Uint8Array(wavArrayBuffer)
        
        // Save to file system via Electron
        const filename = await window.electronAPI.saveAudioFile(wavBuffer)
        
        // Stop all tracks
        mediaRecorder.stream.getTracks().forEach(track => track.stop())
        
        isRecording = false
        console.log('🎤 Audio recording stopped, saved as:', filename)
        resolve(filename)
        
      } catch (error) {
        console.error('❌ Error stopping recording:', error)
        isRecording = false
        reject(error)
      }
    }
    
    mediaRecorder.stop()
  })
}

async function convertToWav(arrayBuffer) {
  try {
    console.log('🎤 Converting audio to WAV format...')
    
    // Simple conversion to WAV format
    // In a real implementation, you'd use a proper audio conversion library
    const audioContext = new (window.AudioContext || window.webkitAudioContext)()
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
    
    console.log('🎤 Audio decoded, duration:', audioBuffer.duration, 'channels:', audioBuffer.numberOfChannels)
    
    // Create WAV file
    const wavBuffer = audioBufferToWav(audioBuffer)
    return new Blob([wavBuffer], { type: 'audio/wav' })
  } catch (error) {
    console.error('❌ Error converting to WAV:', error)
    throw error
  }
}

function audioBufferToWav(buffer) {
  const length = buffer.length
  const numberOfChannels = buffer.numberOfChannels
  const sampleRate = buffer.sampleRate
  const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2)
  const view = new DataView(arrayBuffer)
  
  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i))
    }
  }
  
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + length * numberOfChannels * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, numberOfChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * numberOfChannels * 2, true)
  view.setUint16(32, numberOfChannels * 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, length * numberOfChannels * 2, true)
  
  // Audio data
  let offset = 44
  for (let i = 0; i < length; i++) {
    for (let channel = 0; channel < numberOfChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true)
      offset += 2
    }
  }
  
  return arrayBuffer
}

//ASR CAPTURE TRIGGER TO BACKEND
export async function sendListenRequest(filename, sessionId = null) {
  const res = await fetch('http://localhost:8000/asr', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ sessionId }),
  })

  if (!res.ok) {
    throw new Error(`ASR request failed with status ${res.status}`)
  }

  const data = await res.json()
  
  // Store event if session is active
  if (sessionId) {
    try {
      await api.storeEvent(sessionId, 'asr', data.message || 'Audio transcribed', {
        filename,
        timestamp: new Date().toISOString(),
        type: 'audio',
        source: 'asr'
      })
      console.log('ASR event stored for session:', sessionId)
    } catch (eventError) {
      console.warn('Failed to store ASR event:', eventError.message)
      // Don't fail the ASR if event storage fails
    }
  }

  return data.message
}

