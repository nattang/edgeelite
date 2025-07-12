import mic from 'mic'
import fs from 'fs'
import path from 'path'
import { app } from 'electron'
import fetch from 'node-fetch'
import FormData from 'form-data'

let micInstance
let micInputStream
let outputFilePath

export async function startListening() {
  if (micInstance) return 'Already listening'

  outputFilePath = path.join(app.getPath('userData'), 'mic_capture.wav')

  micInstance = mic({
    rate: '16000',
    channels: '1',
    debug: false,
    fileType: 'wav',
  })

  const micStream = micInstance.getAudioStream()
  micInputStream = micStream.pipe(fs.createWriteStream(outputFilePath))

  micStream.on('error', (err) => {
    console.error('Mic error:', err)
  })

  micInstance.start()
  return 'Listening started'
}

export async function stopListeningAndSend() {
  if (!micInstance) return 'Mic not running'

  return new Promise((resolve, reject) => {
    micInputStream.on('close', async () => {
      try {
        const res = await sendAudioToBackend(outputFilePath)
        resolve(res)
      } catch (err) {
        reject('Failed to send audio: ' + err)
      }
    })

    micInstance.stop()
    micInstance = null
  })
}


async function sendAudioToBackend(filePath) {
  const formData = new FormData()
  formData.append('file', fs.createReadStream(filePath))

  const res = await fetch('http://localhost:5000/asr', {
    method: 'POST',
    body: formData,
  })

  const data = await res.json()
  return data
}
