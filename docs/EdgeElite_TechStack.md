# Tech Stack for EdgeElite Hackathon

## 🖥️ Language & Runtimes
- **Node.js v18+** – for your Electron/Next.js front-end and glue code  
- **Python 3.9+** – for model conversion, FAISS indexing, and any inference scripts  

## 🎨 Front-End UI
- **Next.js v13 (App Router)** with **React 18**  
- **Electron v25** – desktop shell (Windows/macOS)  
- **Tailwind CSS v3** – utility-first styling for “glassmorphism”  
- *(Optional)* **shadcn/ui** – ready-made cards, buttons, modals  

## 💾 Local Storage & Indexing
- **SQLite** via `better-sqlite3` (npm) – persisting journals, profiles, logs  
- **FAISS (faiss-cpu)** (Python) – in-memory vector store for embeddings  

## 🤖 On-Device Inference
- **Qualcomm SNPE SDK v2.x** or **ONNX Runtime v1.x** – edge-optimized model runtime  
- **Quantized LLM** (e.g., LLaMA-7B INT8) for summarization & tutoring  
- **MiniLM-INT8** (or similar) for embeddings  

## 🖼️ OCR & CV
- **ONNX-TrOCR** (exported from Hugging Face) via ONNX Runtime  
- *Fallback:* **Tesseract.js** for quick OCR  

## 🎤 Speech & Audio
- **Qualcomm Conformer ASR** (ONNX) – sub-300 ms on X-Elite  
- **webrtcvad** (Python or Node binding) – voice-activity detection  

## 🔗 Embeddings & Retrieval
- Compute embeddings with **onnxruntime-node** (or Python ONNX Runtime)  
- Index & query with **FAISS** (Python)  

## 🔄 Orchestration & Queues
- **Python asyncio** – for inference pipelines  
- *(Alternative)* **BullMQ** / **bee-queue** (Node.js) – job-queue patterns  

## 📦 Packaging & Tooling
- **Electron Forge** or **Electron Builder** – build Windows/macOS executables  
- **Yarn** or **npm** – dependency management  
- **VS Code** with Mermaid plugin – for diagrams & quick edits  

*Generated on July 11, 2025 by EdgeElite Hackathon Planning Assistant.*
