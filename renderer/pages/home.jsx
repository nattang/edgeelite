import React from 'react'
import Head from 'next/head'
import Link from 'next/link'
import Image from 'next/image'
import { startRecording, stopRecording, sendListenRequest } from '../lib/audio'
import { sendCaptureRequest } from '../lib/capture'
import { api } from '../lib/api'

export default function HomePage() {
  const [message, setMessage] = React.useState('No message found')
  const [inputValue, setInputValue] = React.useState('')
  const [screenshot, setScreenshot] = React.useState(null)
  const [isCapturing, setIsCapturing] = React.useState(false)
  const [isListening, setIsListening] = React.useState(false)
  const [sessionId, setSessionId] = React.useState(null)
  const [isSessionActive, setIsSessionActive] = React.useState(false)
  const [isSummarizing, setIsSummarizing] = React.useState(false)

  React.useEffect(() => {
    window.ipc.on('message', (msg) => {
      setMessage(msg)
    })
  }, [])

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  const startSession = () => {
    const newSessionId = generateSessionId()
    setSessionId(newSessionId)
    setIsSessionActive(true)
    setMessage(`Session started: ${newSessionId}`)
  }

  const endSession = () => {
    setSessionId(null)
    setIsSessionActive(false)
    setMessage('Session ended')
  }

  const handleSend = () => {
    window.ipc.send('message', 'Hello')
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
        setMessage(`Screenshot captured successfully at ${result.timestamp}. Saved to: ${result.filePath}`)
        
        // Send to OCR and store event
        try {
          await sendCaptureRequest(result.filePath, sessionId)
          setMessage(`Screenshot captured and processed. Event stored for session: ${sessionId}`)
        } catch (ocrError) {
          console.warn('OCR processing failed:', ocrError.message)
          setMessage(`Screenshot captured but OCR processing failed: ${ocrError.message}`)
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
      setMessage('Stopped. Processing...')

      try {
        console.log('Stopping audio recording...')
        const filename = await stopRecording()
        console.log('Audio file saved as:', filename)
        const result = await sendListenRequest(filename, sessionId)
        setMessage(result)
      } catch (err) {
        console.error('Error in sendListenRequest:', err)
        setMessage('Failed to process audio')
      }
    }
  }

  const handleSummarize = async () => {
    if (!sessionId) {
      setMessage('Please start a session first')
      return
    }
    
    setIsSummarizing(true)
    setMessage('Getting context and generating summary...')
    
    try {
      // Get context from storage
      const contextData = await api.getContext(sessionId, 10)
      
      // Query LLM with context
      const llmResponse = await api.queryLLM(
        sessionId, 
        "Summarize what I've been working on", 
        contextData.context || []
      )
      
      setMessage(llmResponse.response || 'Summary generated')
    } catch (error) {
      setMessage(`Summarize failed: ${error.message}`)
    } finally {
      setIsSummarizing(false)
    }
  }

  return (
    <>
      <Head>
        <title>Home - Nextron (basic-lang-javascript)</title>
      </Head>

      <div className="p-6 max-w-md mx-auto">
        {/* Session Management */}
        <div className="mb-4 p-3 bg-gray-100 rounded">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Session Status:</span>
            <span className={`text-sm px-2 py-1 rounded ${
              isSessionActive ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-600'
            }`}>
              {isSessionActive ? 'Active' : 'Inactive'}
            </span>
          </div>
          {sessionId && (
            <div className="text-xs text-gray-600 mb-2">
              ID: {sessionId}
            </div>
          )}
          <button
            onClick={isSessionActive ? endSession : startSession}
            className={`w-full py-2 rounded ${
              isSessionActive 
                ? 'bg-red-600 text-white hover:bg-red-700' 
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isSessionActive ? 'End Session' : 'Start Session'}
          </button>
        </div>

        <div className="flex mb-4">
          <input
            type="text"
            placeholder="Type your message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="flex-grow border border-gray-300 rounded-l px-4 py-2 focus:outline-none focus:ring-blue-500"
          />
          <button
            onClick={handleSend}
            className="bg-blue-600 text-white px-4 py-2 rounded-r hover:bg-blue-700 focus:outline-none focus:ring-blue-500"
          >
            ask
          </button>
        </div>

        <div className="flex space-x-4 mb-4">
          <button
            className="flex-1 py-2 bg-gray-200 rounded hover:bg-gray-300"
            onClick={handleListen}
            disabled={!isSessionActive}
          >
            {isListening ? 'Stop' : 'Listen'}
          </button>
          <button 
            onClick={handleCapture}
            disabled={isCapturing || !isSessionActive}
            className={`flex-1 py-2 rounded ${
              isCapturing || !isSessionActive
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
          >
            {isCapturing ? 'Capturing...' : 'Capture'}
          </button>
          <button 
            onClick={handleSummarize}
            disabled={!isSessionActive || isSummarizing}
            className={`flex-1 py-2 rounded ${
              !isSessionActive || isSummarizing
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-purple-600 text-white hover:bg-purple-700'
            }`}
          >
            {isSummarizing ? 'Summarizing...' : 'Summarize'}
          </button>
        </div>

        {screenshot && (
          <div className="mb-4">
            <h3 className="text-lg font-semibold mb-2">Screenshot Preview:</h3>
            <img 
              src={screenshot} 
              alt="Screenshot" 
              className="w-full h-auto border border-gray-300 rounded"
            />
          </div>
        )}

        <p className="text-gray-700">
          {message}
        </p>
      </div>
    </>
  )
}
