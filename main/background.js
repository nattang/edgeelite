import path from 'path'
import { app, ipcMain, desktopCapturer, BrowserWindow } from 'electron'
import serve from 'electron-serve'
import { createWindow } from './helpers'
import fs from 'fs'
import os from 'os'
import { startRecording, stopRecording } from './record.js'

const isProd = process.env.NODE_ENV === 'production'

if (isProd) {
  serve({ directory: 'app' })
} else {
  app.setPath('userData', `${app.getPath('userData')} (development)`)
}

;(async () => {
  await app.whenReady()

  const mainWindow = createWindow('main', {
    width: 800,
    height: 200,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  })

  if (isProd) {
    await mainWindow.loadURL('app://./home')
  } else {
    const port = process.argv[2]
    await mainWindow.loadURL(`http://localhost:${port}/home`)
    mainWindow.webContents.openDevTools()
  }
})()

app.on('window-all-closed', () => {
  app.quit()
})

ipcMain.on('message', async (event, arg) => {
  event.reply('message', `${arg} World!`)
})

ipcMain.handle('take-screenshot', async () => {
  try {
    // Get all screen sources
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: 1920, height: 1080 }
    })

    if (sources.length === 0) {
      throw new Error('No screen sources found')
    }

    // Get the primary display
    const primarySource = sources.find(source => source.display_id === '0:0') || sources[0]
    
    // Convert the thumbnail to base64
    const image = primarySource.thumbnail.toDataURL()
    
    // Save image to captures folder
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    const capturesDir = path.join(os.homedir(), 'EdgeElite', 'captures')
    
    // Create captures directory if it doesn't exist
    if (!fs.existsSync(capturesDir)) {
      fs.mkdirSync(capturesDir, { recursive: true })
    }
    
    const imagePath = path.join(capturesDir, `screenshot-${timestamp}.png`)
    
    // Convert base64 to buffer and save
    const base64Data = image.replace(/^data:image\/png;base64,/, '')
    const buffer = Buffer.from(base64Data, 'base64')
    fs.writeFileSync(imagePath, buffer)
    
    return {
      success: true,
      image: image,
      timestamp: new Date().toISOString(),
      filePath: imagePath
    }
  } catch (error) {
    console.error('Screenshot error:', error)
    return {
      success: false,
      error: error.message
    }
  }
})

ipcMain.handle('audio:start', async () => {
  return await startRecording()
})

ipcMain.handle('audio:stop', async () => {
  console.log('Stopping audio recording...')
  const filename = await stopRecording()
  return filename
})
