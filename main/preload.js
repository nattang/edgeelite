import { contextBridge, ipcRenderer, desktopCapturer } from 'electron'


const handler = {
  send(channel, value) {
    ipcRenderer.send(channel, value)
  },
  on(channel, callback) {
    const subscription = (_event, ...args) => callback(...args)
    ipcRenderer.on(channel, subscription)

    return () => {
      ipcRenderer.removeListener(channel, subscription)
    }
  },
  invoke(channel, value) {
    return ipcRenderer.invoke(channel, value)
  },
}

contextBridge.exposeInMainWorld('ipc', handler)

contextBridge.exposeInMainWorld('electronAPI', {
  desktopCapturer,
  takeScreenshot: () => ipcRenderer.invoke('take-screenshot'),
})

contextBridge.exposeInMainWorld('audio', {
  startListening: () => ipcRenderer.invoke('audio:start'),
  stopListening: () => ipcRenderer.invoke('audio:stop')
})
