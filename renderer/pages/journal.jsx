import React from 'react'
import Head from 'next/head'
import Link from 'next/link'
import { startRecording, stopRecording, sendListenRequest } from '../lib/audio'
import { sendCaptureRequest } from '../lib/capture'
import { api } from '../lib/api'

export default function JournalPage() {
  const [sessionId, setSessionId] = React.useState(null)
  const [isSessionActive, setIsSessionActive] = React.useState(false)
  const [journalEntry, setJournalEntry] = React.useState(null)
  const [isProcessing, setIsProcessing] = React.useState(false)
  const [showRelatedModal, setShowRelatedModal] = React.useState(false)
  const [message, setMessage] = React.useState('Welcome to EdgeElite Journal')
  const [isCapturing, setIsCapturing] = React.useState(false)
  const [isListening, setIsListening] = React.useState(false)
  const [screenshot, setScreenshot] = React.useState(null)

  // Session management functions
  const generateSessionId = () => {
    return `journal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  const startSession = () => {
    const newSessionId = generateSessionId()
    setSessionId(newSessionId)
    setIsSessionActive(true)
    setJournalEntry(null)
    setMessage(`Journal session started: ${newSessionId}`)
  }

  const endSession = async () => {
    if (!sessionId) return

    setIsSessionActive(false)
    setIsProcessing(true)
    setMessage('Ending session and generating journal entry...')

    try {
      // End session and start journal processing
      await api.endSession(sessionId)
      setMessage('Processing your journal entry...')

      // Poll for journal results
      const result = await api.pollJournal(sessionId)
      setJournalEntry(result)
      setMessage('Journal entry ready!')
    } catch (error) {
      console.error('Journal processing failed:', error)
      setMessage(`Journal processing failed: ${error.message}`)
    } finally {
      setIsProcessing(false)
    }
  }

  // Screenshot capture function
  const handleCapture = async () => {
    if (!isSessionActive) {
      setMessage('Please start a session first')
      return
    }

    setIsCapturing(true)
    try {
      const result = await window.electronAPI.takeScreenshot()
      if (result.success) {
        setScreenshot(result.image)
        setMessage(`Screenshot captured at ${result.timestamp}`)
        
        // Send to OCR and store event
        try {
          await sendCaptureRequest(result.filePath, sessionId)
          setMessage(`Screenshot processed and stored for session: ${sessionId}`)
        } catch (ocrError) {
          console.warn('OCR processing failed:', ocrError.message)
          setMessage(`Screenshot captured but OCR failed: ${ocrError.message}`)
        }
      } else {
        setMessage(`Screenshot failed: ${result.error}`)
      }
    } catch (error) {
      setMessage(`Screenshot error: ${error.message}`)
    } finally {
      setIsCapturing(false)
    }
  }

  // Audio recording function
  const handleListen = async () => {
    if (!isSessionActive) {
      setMessage('Please start a session first')
      return
    }

    if (!isListening) {
      setIsListening(true)
      setMessage('Listening...')
      try {
        await startRecording()
      } catch (error) {
        setMessage(`Failed to start recording: ${error.message}`)
        setIsListening(false)
      }
    } else {
      setIsListening(false)
      setMessage('Stopped. Processing audio...')

      try {
        const filename = await stopRecording()
        const result = await sendListenRequest(filename, sessionId)
        setMessage(`Audio processed: ${result}`)
      } catch (err) {
        console.error('Audio processing error:', err)
        setMessage('Failed to process audio')
      }
    }
  }

  return (
    <>
      <Head>
        <title>Journal - EdgeElite</title>
      </Head>

      <div className="p-6 max-w-4xl mx-auto">
        {/* Navigation */}
        <div className="mb-6">
          <Link href="/home" className="text-blue-600 hover:text-blue-800">
            ← Back to Home
          </Link>
        </div>

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">EdgeElite Journal</h1>
          <p className="text-gray-600">Capture your thoughts and get personalized guidance powered by AI</p>
        </div>

        {/* Session Controls */}
        <div className="bg-white border rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Session Management</h2>
          
          {/* Session Status */}
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-gray-700">Session Status:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                isSessionActive ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
              }`}>
                {isSessionActive ? 'Active' : 'Inactive'}
              </span>
            </div>
            
            {sessionId && (
              <div className="text-sm text-gray-500">
                Session ID: {sessionId}
              </div>
            )}
          </div>
          
          {/* Control Buttons */}
          <div className="flex gap-3 mb-4">
            <button
              onClick={isSessionActive ? endSession : startSession}
              disabled={isProcessing}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                isSessionActive 
                  ? 'bg-red-600 text-white hover:bg-red-700' 
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isProcessing ? 'Processing...' : isSessionActive ? 'End Session & Generate Journal' : 'Start New Session'}
            </button>
          </div>

          {/* Capture Buttons */}
          {isSessionActive && (
            <div className="flex gap-3">
              <button
                onClick={handleCapture}
                disabled={isCapturing}
                className={`px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors ${
                  isCapturing ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isCapturing ? 'Capturing...' : '📸 Take Screenshot'}
              </button>
              
              <button
                onClick={handleListen}
                disabled={false}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  isListening 
                    ? 'bg-red-600 text-white hover:bg-red-700' 
                    : 'bg-purple-600 text-white hover:bg-purple-700'
                }`}
              >
                {isListening ? '🛑 Stop Recording' : '🎤 Start Recording'}
              </button>
            </div>
          )}
        </div>

        {/* Status Message */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <p className="text-blue-800">{message}</p>
        </div>

        {/* Screenshot Preview */}
        {screenshot && (
          <div className="bg-white border rounded-lg p-6 mb-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-3">Latest Screenshot</h3>
            <img 
              src={screenshot} 
              alt="Screenshot" 
              className="w-full max-w-md h-auto border border-gray-300 rounded"
            />
          </div>
        )}

        {/* Journal Entry Display */}
        {journalEntry && !journalEntry.error && (
          <div className="bg-white border rounded-lg p-6 shadow-sm">
            <h2 className="text-2xl font-semibold mb-4 text-gray-900">Journal Entry</h2>
            
            <div className="prose max-w-none">
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 mb-4">
                <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">
                  {journalEntry.summary_action}
                </p>
              </div>
            </div>
            
            {/* Related Memory Chip */}
            {journalEntry.related_memory && (
              <div className="mt-6">
                <button
                  onClick={() => setShowRelatedModal(true)}
                  className="inline-flex items-center px-4 py-2 rounded-full text-sm bg-blue-100 text-blue-800 hover:bg-blue-200 transition-colors"
                >
                  🔗 View Related Memory
                </button>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {journalEntry && journalEntry.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-red-800 mb-2">Processing Error</h3>
            <p className="text-red-700">{journalEntry.error}</p>
          </div>
        )}

        {/* Related Memory Modal */}
        {showRelatedModal && journalEntry?.related_memory && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-96 overflow-y-auto">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">Related Memory</h3>
              <div className="bg-gray-50 rounded-lg p-4 mb-4">
                <p className="text-gray-700 whitespace-pre-wrap">
                  {journalEntry.related_memory}
                </p>
              </div>
              <div className="text-sm text-gray-500 mb-4">
                This memory was found through semantic similarity to your current session.
              </div>
              <div className="flex justify-end">
                <button
                  onClick={() => setShowRelatedModal(false)}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
} 