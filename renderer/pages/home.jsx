import React from 'react'
import Head from 'next/head'
import Link from 'next/link'
import Image from 'next/image'
import { startRecording, stopRecording, sendListenRequest } from '../lib/audio'

import { sendCaptureRequest } from '../lib/capture'

export default function HomePage() {
  const [message, setMessage] = React.useState('No message found')
  const [inputValue, setInputValue] = React.useState('')
  const [screenshot, setScreenshot] = React.useState(null)
  const [isCapturing, setIsCapturing] = React.useState(false)
  const [isListening, setIsListening] = React.useState(false)

  React.useEffect(() => {
    window.ipc.on('message', (msg) => {
      setMessage(msg)
    })
  }, [])

  const handleSend = () => {
    window.ipc.send('message', 'Hello')
  }

  const handleCapture = async () => {
    setIsCapturing(true)
    try {
      const result = await window.electronAPI.takeScreenshot()
      if (result.success) {
        setScreenshot(result.image)
        setMessage(`Screenshot captured successfully at ${result.timestamp}. Saved to: ${result.filePath}`)
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
        const result = await sendListenRequest(filename)
        setMessage(result)
      } catch (err) {
        console.error('Error in sendListenRequest:', err)
        setMessage('Failed to process audio')
      }
    }
  }

  return (
    <>
      <Head>
        <title>Home - Nextron (basic-lang-javascript)</title>
      </Head>

      <div className="p-6 max-w-md mx-auto">
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
          >
            {isListening ? 'Stop' : 'Listen'}
          </button>
          <button 
            onClick={handleCapture}
            disabled={isCapturing}
            className={`flex-1 py-2 rounded ${
              isCapturing 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
          >
            {isCapturing ? 'Capturing...' : 'Capture'}
          </button>
          <button className="flex-1 py-2 bg-gray-200 rounded hover:bg-gray-300">
            Write
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
