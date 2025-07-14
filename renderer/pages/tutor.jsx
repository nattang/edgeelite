import React from 'react'
import Head from 'next/head'
import Link from 'next/link'
import { startRecording, stopRecording, sendListenRequest } from '../lib/audio'
import { sendCaptureRequest } from '../lib/capture'
import { api } from '../lib/api'

export default function TutorPage() {
  const [sessionId, setSessionId] = React.useState(null)
  const [isSessionActive, setIsSessionActive] = React.useState(false)
  const [message, setMessage] = React.useState('Welcome to EdgeElite Tutor')
  const [isCapturing, setIsCapturing] = React.useState(false)
  const [isListening, setIsListening] = React.useState(false)
  const [screenshot, setScreenshot] = React.useState(null)

  const generateSessionId = () => {
    return `tutor_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  const startSession = () => {
    const newSessionId = generateSessionId()
    setSessionId(newSessionId)
    setIsSessionActive(true)
    setMessage(`Tutor session started: ${newSessionId}`)
  }

  const endSession = () => {
    setIsSessionActive(false)
    setMessage('Tutor session ended.')
  }

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
        await sendCaptureRequest(result.filePath, sessionId)
      } else {
        setMessage(`Screenshot failed: ${result.error}`)
      }
    } catch (error) {
      setMessage(`Screenshot error: ${error.message}`)
    } finally {
      setIsCapturing(false)
    }
  }

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
        setMessage('Failed to process audio')
      }
    }
  }

  return (
    <>
      <Head>
        <title>Tutor - EdgeElite</title>
      </Head>

      <div className="p-6 max-w-4xl mx-auto">
        <div className="mb-6">
          <Link href="/home" className="text-blue-600 hover:text-blue-800">
            ← Back to Home
          </Link>
        </div>

        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">EdgeElite Tutor</h1>
          <p className="text-gray-600">Use AI assistance to guide your learning</p>
        </div>

        <div className="bg-white border rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Session Management</h2>

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
              <div className="text-sm text-gray-500">Session ID: {sessionId}</div>
            )}
          </div>

          <div className="flex gap-3 mb-4">
            <button
              onClick={isSessionActive ? endSession : startSession}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                isSessionActive
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {isSessionActive ? 'End Session' : 'Start Session'}
            </button>
          </div>

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

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <p className="text-blue-800">{message}</p>
        </div>

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
      </div>
    </>
  )
}