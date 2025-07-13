import { api } from './api'

// Voice query management for context recall
export class VoiceQueryManager {
  constructor() {
    this.mediaRecorder = null
    this.audioChunks = []
    this.isListening = false
    this.sessionId = null
    this.onResponseCallback = null
    this.currentStream = null
    this.recordingTimeout = null
  }

  async startContinuousListening(sessionId, onResponse) {
    this.sessionId = sessionId
    this.onResponseCallback = onResponse

    try {
      // Get user media permission
      this.currentStream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      // Start recording in chunks
      this.startRecordingChunk()
      
      this.isListening = true
      console.log('🎤 Voice query manager started')
      
    } catch (error) {
      console.error('Voice listening error:', error)
      throw error
    }
  }

  startRecordingChunk() {
    if (!this.currentStream || !this.isListening) return

    this.audioChunks = []
    this.mediaRecorder = new MediaRecorder(this.currentStream)
    
    this.mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) {
        this.audioChunks.push(event.data)
      }
    })

    this.mediaRecorder.addEventListener('stop', () => {
      this.processVoiceQuery()
    })

    // Record for 30 seconds
    this.mediaRecorder.start()
    
    // Schedule next recording chunk
    this.recordingTimeout = setTimeout(() => {
      if (this.isListening && this.mediaRecorder) {
        this.mediaRecorder.stop()
        
        // Start next chunk after processing
        setTimeout(() => {
          this.startRecordingChunk()
        }, 1000)
      }
    }, 30000)
  }

  async processVoiceQuery() {
    if (this.audioChunks.length === 0) return

    try {
      const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' })
      
      // Convert to WAV and save (reuse existing audio processing)
      const timestamp = Date.now()
      const filename = `query-${timestamp}.wav`
      
      console.log('🎤 Processing voice query, saving as:', filename)
      
      // Save audio file using Electron API
      await window.electronAPI.saveAudioFile(audioBlob, filename)
      
      // Send to ASR for transcription
      const asrResponse = await fetch('http://localhost:8000/asr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: this.sessionId })
      })

      if (asrResponse.ok) {
        const asrResult = await asrResponse.json()
        const transcript = asrResult.message
        
        console.log('🎤 Transcription result:', transcript)

        // Check if this looks like a recall query
        if (this.isRecallQuery(transcript)) {
          console.log('🎯 Recall query detected, processing...')
          await this.handleRecallQuery(transcript)
        } else {
          console.log('🎤 Not a recall query, continuing to listen...')
        }
      }
    } catch (error) {
      console.error('Voice query processing error:', error)
    }
  }

  isRecallQuery(transcript) {
    const recallKeywords = [
      'what did i say',
      'remind me',
      'what was mentioned',
      'what was said',
      'tell me about',
      'recall',
      'edgeelite',
      'what did we discuss',
      'what happened with'
    ]
    
    const lowerTranscript = transcript.toLowerCase()
    return recallKeywords.some(keyword => lowerTranscript.includes(keyword))
  }

  async handleRecallQuery(transcript) {
    try {
      console.log('🔍 Sending recall query to backend:', transcript)
      
      const response = await api.contextRecall(this.sessionId, transcript)
      
      console.log('🤖 Recall response received:', response)
      
      // Call the callback with the response
      if (this.onResponseCallback) {
        this.onResponseCallback(response)
      }
      
      // Auto-hide response after 10 seconds
      setTimeout(() => {
        if (this.onResponseCallback) {
          this.onResponseCallback(null)
        }
      }, 10000)
      
    } catch (error) {
      console.error('Recall query error:', error)
      
      // Show error response
      if (this.onResponseCallback) {
        this.onResponseCallback({
          answer: "I encountered an error while trying to recall that information.",
          sources: [],
          confidence: 0.0,
          session_id: this.sessionId
        })
      }
    }
  }

  stopListening() {
    console.log('🛑 Stopping voice query manager')
    
    this.isListening = false
    
    // Stop current recording
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop()
    }
    
    // Clear timeout
    if (this.recordingTimeout) {
      clearTimeout(this.recordingTimeout)
      this.recordingTimeout = null
    }
    
    // Stop media stream
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop())
      this.currentStream = null
    }
    
    // Clear data
    this.audioChunks = []
    this.mediaRecorder = null
    this.sessionId = null
    this.onResponseCallback = null
  }
}

export default VoiceQueryManager 