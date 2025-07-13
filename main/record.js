import fs from 'fs'
import mic from 'mic'
import path from 'path'

const AUDIO_FILE = path.join(__dirname, 'temp_audio.wav')
let micInstance
let micInputStream

export function startRecording() {
  return new Promise((resolve, reject) => {
    try {
      micInstance = mic({
        rate: '16000',
        channels: '1',
        debug: false,
        // exitOnSilence: 6,
        fileType: 'wav',
      })

      micInputStream = micInstance.getAudioStream()
      const outputFileStream = fs.createWriteStream(AUDIO_FILE)

      micInputStream.pipe(outputFileStream)

      micInputStream.on('startComplete', () => {
        console.log('Mic started')
        resolve('Recording started')
      })

      micInstance.start()
    } catch (err) {
      reject(err)
    }
  })
}

export function stopRecording() {
  return new Promise((resolve, reject) => {
    console.log('Stopping mic...')

    if (micInstance && micInputStream) {
      micInputStream.once('stopComplete', () => {
        console.log('Mic stopped')
        resolve('temp_audio.wav')
      })

      micInstance.stop()
    } else {
      reject(new Error('Mic not started'))
    }
  })
}

