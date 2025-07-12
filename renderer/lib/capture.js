export async function sendCaptureRequest() {
  try {
    const res = await fetch('http://localhost:3000/capture', {
      method: 'POST',
       headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filename: 'dummy_screenshot.png' })

    })

    const data = await res.json();
    return data.message;
  } catch (err) {
    console.error('Capture request failed:', err);
    throw err;
  }
}
