import React from 'react'
import Head from 'next/head'
import Link from 'next/link'
import Image from 'next/image'
import { startRecording, stopRecording, sendListenRequest } from '../lib/audio'
import { sendCaptureRequest } from '../lib/capture'
import { api } from '../lib/api'


export default function HomePage() {
  const [activeTab, setActiveTab] = React.useState('journal')
  const [message, setMessage] = React.useState('No message found')
  const [inputValue, setInputValue] = React.useState('')
  const [screenshot, setScreenshot] = React.useState(null)
  const [isCapturing, setIsCapturing] = React.useState(false)
  const [isListening, setIsListening] = React.useState(false)
  const [sessionId, setSessionId] = React.useState(null)
  const [isSessionActive, setIsSessionActive] = React.useState(false)
  const [isSummarizing, setIsSummarizing] = React.useState(false)

  // Journal state
  const [journalEntry, setJournalEntry] = React.useState(null)
  const [journalEntries, setJournalEntries] = React.useState([])
  const [isProcessing, setIsProcessing] = React.useState(false)
  const [showRelatedModal, setShowRelatedModal] = React.useState(false)
  const [selectedRelatedMemory, setSelectedRelatedMemory] = React.useState(null)


  
  // Recall tab specific state
  const [recallSessionId, setRecallSessionId] = React.useState(null)
  const [isRecallSessionActive, setIsRecallSessionActive] = React.useState(false)
  const [recallQuery, setRecallQuery] = React.useState('')
  const [recallResponse, setRecallResponse] = React.useState(null)
  const [isRecallProcessing, setIsRecallProcessing] = React.useState(false)
  const [assistantResponse, setAssistantResponse] = React.useState(null)

  React.useEffect(() => {
    window.ipc.on('message', (msg) => {
      setMessage(msg)
    })
  }, [])

  // Load journal entries when component mounts or when journal tab is selected
  const loadJournalEntries = async () => {
    try {
      const response = await api.getJournalEntries()
      setJournalEntries(response.entries || [])
    } catch (error) {
      console.error('Failed to load journal entries:', error)
    }
  }

  React.useEffect(() => {
    loadJournalEntries()
  }, [])

  React.useEffect(() => {
    if (activeTab === 'journal') {
      loadJournalEntries()
    }
  }, [activeTab])

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  const startSession = async () => {
    const newSessionId = generateSessionId()
    setSessionId(newSessionId)
    setIsSessionActive(true)
    
    // Auto-start recording when session starts
    try {
      await startRecording()
      setIsListening(true)
      setMessage(`Session started: ${newSessionId} - Recording...`)
    } catch (error) {
      console.error('Failed to start recording:', error)
      setMessage(`Session started: ${newSessionId} - Recording failed to start: ${error.message}`)
      setIsListening(false)
    }
  }

  const endSession = async () => {
    if (!sessionId) return
    
    setIsSessionActive(false)
    setIsProcessing(true)
    
    try {
      // Stop recording and process audio first (if recording was active)
      if (isListening) {
        setMessage('Stopping recording and processing audio...')
        try {
          console.log('Stopping audio recording...')
          const filename = await stopRecording()
          console.log('Audio file saved as:', filename)
          const result = await sendListenRequest(filename, sessionId)
          console.log('Audio processing result:', result)
          setIsListening(false)
        } catch (audioError) {
          console.error('Error processing audio:', audioError)
          setMessage('Audio processing failed, but continuing with journal...')
          setIsListening(false)
        }
      }
      
      // End session and start journal processing
      await api.endSession(sessionId)
      
      // Poll for journal results
      const result = await api.pollJournal(sessionId)
      setJournalEntry(result)
      setMessage('Journal entry generated successfully')
      
      // Refresh journal entries list
      await loadJournalEntries()
    } catch (error) {
      console.error('Journal processing failed:', error)
      setMessage('Failed to generate journal entry')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleSend = () => {
    window.ipc.send('message', 'Hello')
  }

  const handleCapture = async () => {
    if (!isSessionActive) {
      setMessage('Please start a session first')
      return
    }

    if (isCapturing) return

    setIsCapturing(true)
    setMessage('Taking screenshot...')

    try {
      const result = await window.electronAPI.takeScreenshot()
      console.log('Screenshot taken:', result)
      setScreenshot(result)
      
      // Send capture request to backend
      await sendCaptureRequest(result.filePath, sessionId)
      setMessage('Screenshot processed and stored')
    } catch (error) {
      console.error('Capture error:', error)
      setMessage(`Capture error: ${error.message}`)
    } finally {
      setIsCapturing(false)
    }
  }



  const handleSummarize = async () => {
    if (!sessionId) {
      setMessage('No active session to summarize')
      return
    }

    setIsSummarizing(true)
    setMessage('Generating summary...')

    try {
      const response = await api.queryLLM(sessionId, 'Summarize this session')
      setMessage(`Summary: ${response.response}`)
    } catch (error) {
      console.error('Summary error:', error)
      setMessage(`Summary error: ${error.message}`)
    } finally {
      setIsSummarizing(false)
    }
  }



  // Recall tab session functions
  const startRecallSession = () => {
    const newSessionId = generateSessionId()
    setRecallSessionId(newSessionId)
    setIsRecallSessionActive(true)
    setMessage(`Recall session started: ${newSessionId}`)
  }

  const endRecallSession = () => {
    setIsRecallSessionActive(false)
    setRecallSessionId(null)
    setRecallResponse(null)
    setAssistantResponse(null)
    setMessage('Recall session ended')
  }

  const handleRecallCapture = async () => {
    if (!isRecallSessionActive) {
      setMessage('Please start a recall session first')
      return
    }

    if (isCapturing) return

    setIsCapturing(true)
    setMessage('Taking screenshot...')

    try {
      const result = await window.electronAPI.takeScreenshot()
      console.log('Screenshot taken:', result)
      setScreenshot(result)
      
      // Send capture request to backend
      await sendCaptureRequest(result.filePath, recallSessionId)
      setMessage('Screenshot processed and stored')
    } catch (error) {
      console.error('Capture error:', error)
      setMessage(`Capture error: ${error.message}`)
    } finally {
      setIsCapturing(false)
    }
  }

  const handleRecallQuery = async (e) => {
    e.preventDefault()
    
    if (!recallQuery.trim()) {
      setMessage('Please enter a query')
      return
    }

    if (!recallSessionId) {
      setMessage('Please start a recall session first')
      return
    }

    setIsRecallProcessing(true)
    setMessage('Processing recall query...')

    try {
      const response = await api.contextRecall(recallSessionId, recallQuery)
      setRecallResponse(response)
      setAssistantResponse(response) // Show in assistant bubble
      setMessage('Recall query processed successfully')
      setRecallQuery('') // Clear the query after successful submission
    } catch (error) {
      console.error('Recall query error:', error)
      setMessage(`Recall query error: ${error.message}`)
    } finally {
      setIsRecallProcessing(false)
    }
  }

  const renderJournalTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Journal Session</h2>
        
        {/* Session Status */}
        <div className="bg-gray-100 rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between">
            <span className="font-medium">Session Status:</span>
            <span className={`px-3 py-1 rounded-full text-sm ${
              isSessionActive ? 'bg-green-200 text-green-800' : 'bg-gray-200'
            }`}>
              {isSessionActive ? 'Active' : 'Inactive'}
            </span>
          </div>
          {sessionId && (
            <div className="text-sm text-gray-600 mt-2">
              Session ID: {sessionId}
            </div>
          )}
          {isSessionActive && (
            <div className="flex items-center mt-2">
              <span className="text-sm font-medium mr-2">Recording:</span>
              <span className={`px-2 py-1 rounded-full text-xs ${
                isListening ? 'bg-red-200 text-red-800' : 'bg-yellow-200 text-yellow-800'
              }`}>
                {isListening ? '🔴 Recording' : '⚠️ Not Recording'}
              </span>
            </div>
          )}
        </div>

        {/* Control Buttons */}
        <div className="flex gap-3 mb-4">
          <button
            onClick={isSessionActive ? endSession : startSession}
            disabled={isProcessing}
            className={`px-4 py-2 rounded-lg font-medium ${
              isSessionActive
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isProcessing ? 'Processing...' : isSessionActive ? 'End Session & Create Journal' : 'Start Session & Recording'}
          </button>

          <button
            onClick={handleCapture}
            disabled={!isSessionActive || isCapturing}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
          >
            {isCapturing ? 'Capturing...' : '📸 Screenshot'}
          </button>
        </div>

        {/* Message Display */}
        <div className="bg-gray-50 rounded p-3 mb-4">
          <p className="text-sm text-gray-700">{message}</p>
        </div>

        {/* Journal Entries Display */}
        <div className="bg-white border rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Journal Entries</h3>
          
          {journalEntries.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No journal entries yet.</p>
              <p className="text-sm mt-2">Start a session and capture some content to create your first entry.</p>
            </div>
          ) : (
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {journalEntries.map((entry, index) => (
                <div key={index} className="border-b pb-4 last:border-b-0">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-gray-600">{entry.date}</span>
                    <span className="text-sm text-gray-500">{entry.time}</span>
                  </div>
                  <div className="prose max-w-none">
                    <p className="whitespace-pre-wrap text-gray-800">{entry.summary_action}</p>
                  </div>
                  {entry.related_memory && (
                    <div className="mt-2">
                      <button
                        onClick={() => {
                          setSelectedRelatedMemory(entry.related_memory)
                          setShowRelatedModal(true)
                        }}
                        className="inline-flex items-center px-3 py-1 rounded-full text-xs bg-blue-100 text-blue-800 hover:bg-blue-200"
                      >
                        🔗 Related Memory
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )

  const renderRecallTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Context Recall</h2>
        <p className="text-gray-600 mb-6">
          Ask EdgeElite to recall information from your previous conversations.
        </p>

        {/* Session Status */}
        <div className="bg-gray-100 rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between">
            <span className="font-medium">Recall Session Status:</span>
            <span className={`px-3 py-1 rounded-full text-sm ${
              isRecallSessionActive ? 'bg-green-200 text-green-800' : 'bg-gray-200'
            }`}>
              {isRecallSessionActive ? 'Active' : 'Inactive'}
            </span>
          </div>
          {recallSessionId && (
            <div className="text-sm text-gray-600 mt-2">
              Session ID: {recallSessionId}
            </div>
          )}
        </div>

        {/* Control Buttons */}
        <div className="flex gap-3 mb-4">
          <button
            onClick={isRecallSessionActive ? endRecallSession : startRecallSession}
            className={`px-4 py-2 rounded-lg font-medium ${
              isRecallSessionActive
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isRecallSessionActive ? 'End Session' : 'Start Session'}
          </button>

          <button
            onClick={handleRecallCapture}
            disabled={!isRecallSessionActive || isCapturing}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
          >
            {isCapturing ? 'Capturing...' : '📸 Screenshot'}
          </button>
        </div>

        {/* Message Display */}
        <div className="bg-gray-50 rounded p-3 mb-4">
          <p className="text-sm text-gray-700">{message}</p>
        </div>

        {/* Chat Interface */}
        <div className="border rounded-lg p-4 mb-4">
          <h3 className="font-medium mb-3">Ask a Question</h3>
          <form onSubmit={handleRecallQuery} className="flex gap-2">
            <input
              type="text"
              value={recallQuery}
              onChange={(e) => setRecallQuery(e.target.value)}
              placeholder="What did I say about Project X?"
              className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isRecallProcessing}
            />
            <button
              type="submit"
              disabled={isRecallProcessing || !recallQuery.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
            >
              {isRecallProcessing ? 'Processing...' : 'Ask'}
            </button>
          </form>
        </div>

        {/* Recall Response */}
        {recallResponse && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <h4 className="font-medium text-blue-900 mb-2">EdgeElite says:</h4>
            <p className="text-blue-800 mb-3">{recallResponse.answer}</p>
            
            {recallResponse.sources && recallResponse.sources.length > 0 && (
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-blue-700 hover:text-blue-800">
                  View Sources ({recallResponse.sources.length})
                </summary>
                <div className="mt-2 space-y-2">
                  {recallResponse.sources.map((source, index) => (
                    <div key={index} className="bg-blue-100 p-2 rounded text-sm">
                      {source.content}
                    </div>
                  ))}
                </div>
              </details>
            )}
            
            <div className="text-xs text-blue-600 mt-2">
              Confidence: {Math.round(recallResponse.confidence * 100)}%
            </div>
          </div>
        )}

        {/* Demo Instructions */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-2">Try these questions:</h4>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>• "What did I say about Project X?"</li>
            <li>• "Remind me about the marketing budget"</li>
            <li>• "What was mentioned about scheduling?"</li>
          </ul>
        </div>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>EdgeElite - AI Assistant</title>
      </Head>

      <div className="max-w-4xl mx-auto p-6">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">EdgeElite</h1>
          <p className="text-gray-600">Your On-Device AI Assistant</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200 mb-6">
          <button
            onClick={() => setActiveTab('journal')}
            className={`px-6 py-3 font-medium text-sm border-b-2 ${
              activeTab === 'journal'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            📖 Journal
          </button>
          <button
            onClick={() => setActiveTab('recall')}
            className={`px-6 py-3 font-medium text-sm border-b-2 ${
              activeTab === 'recall'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            🧠 Recall
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'journal' && renderJournalTab()}
        {activeTab === 'recall' && renderRecallTab()}

        {/* Related Memory Modal */}
        {showRelatedModal && selectedRelatedMemory && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
              <h3 className="text-lg font-semibold mb-3">Related Memory</h3>
              <p className="text-gray-700 mb-4">{selectedRelatedMemory}</p>
              <button
                onClick={() => {
                  setShowRelatedModal(false)
                  setSelectedRelatedMemory(null)
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Close
              </button>
            </div>
          </div>
        )}

        {/* Assistant Response Bubble (for Recall) */}
        {assistantResponse && (
          <div className="fixed bottom-6 right-6 bg-blue-800 text-white p-4 rounded-xl shadow-lg max-w-md z-50">
            <div className="flex justify-between items-start mb-2">
              <span className="font-bold text-blue-200">EdgeElite says:</span>
              <button 
                onClick={() => setAssistantResponse(null)}
                className="text-blue-200 hover:text-white ml-2"
              >
                ×
              </button>
            </div>
            <div className="mb-3">{assistantResponse.answer}</div>
            {assistantResponse.sources && assistantResponse.sources.length > 0 && (
              <div className="text-xs text-blue-200">
                <details>
                  <summary className="cursor-pointer">Sources ({assistantResponse.sources.length})</summary>
                  <div className="mt-2 space-y-1">
                    {assistantResponse.sources.map((source, index) => (
                      <div key={index} className="bg-blue-900 p-2 rounded text-xs">
                        {source.content}
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            )}
            <div className="text-xs text-blue-300 mt-2">
              Confidence: {Math.round(assistantResponse.confidence * 100)}%
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
