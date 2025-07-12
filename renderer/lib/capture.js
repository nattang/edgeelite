export async function sendCaptureRequest() {
  try {
    const res = await fetch('http://localhost:8888/capture', {
      method: 'POST',
    })

    const data = await res.json();
    return data.message;
  } catch (err) {
    console.error('Capture request failed:', err);
    throw err;
  }
}
