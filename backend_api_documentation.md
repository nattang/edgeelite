# EdgeElite Backend API Documentation

This document outlines all the relevant functions and function signatures that are meant to be accessible from the frontend side for the EdgeElite AI Assistant backend.

## Table of Contents

1. [API Endpoints (FastAPI)](#api-endpoints-fastapi)
2. [OCR Module](#ocr-module)
3. [ASR Module](#asr-module)
4. [LLM Service](#llm-service)
5. [Storage & Embedding System](#storage--embedding-system)
6. [Model Management](#model-management)

---

## API Endpoints (FastAPI)

### Core HTTP Endpoints

Located in `backend/main.py` - these are the main endpoints that the frontend can call:

#### `GET /`

```python
@app.get("/")
def read_root() -> dict
```

- **Purpose**: Health check endpoint
- **Returns**: `{"message": "Hello from FastAPI!"}`

#### `POST /asr`

```python
@app.post("/asr")
async def asr() -> dict
```

- **Purpose**: Triggers ASR processing on the most recent audio file
- **Behavior**: Looks for the most recent `.wav` file in `~/EdgeElite/recordings/`
- **Returns**: `{"message": str}` (transcription result or error message)

#### `POST /capture`

```python
@app.post("/capture")
async def capture(data: CaptureRequest) -> dict
```

- **Purpose**: Processes captured screenshots via OCR
- **Input**: `CaptureRequest` object with `filename` field
- **Returns**: `{"message": str}` (processing confirmation)

#### `POST /api/query`

```python
@app.post("/api/query")
async def query_llm(request: QueryRequest) -> dict
```

- **Purpose**: Handles LLM queries with context
- **Input**: `QueryRequest` object
- **Returns**:
  - Success: `{"response": str, "session_id": str}`
  - Error: `{"error": str, "session_id": str}`

#### `POST /api/events`

```python
@app.post("/api/events")
async def store_event(request: EventRequest) -> dict
```

- **Purpose**: Stores events (OCR/ASR results) in the database
- **Input**: `EventRequest` object
- **Returns**:
  - Success: `{"event_id": str, "status": str, "message": str}`
  - Error: `{"error": str, "session_id": str}`

#### `POST /api/context`

```python
@app.post("/api/context")
async def get_context(request: ContextRequest) -> dict
```

- **Purpose**: Retrieves context for a session
- **Input**: `ContextRequest` object
- **Returns**:
  - Success: `{"session_id": str, "context": List[Dict], "count": int, "message": str}`
  - Error: `{"error": str, "session_id": str}`

### Request Models

#### `CaptureRequest`

```python
class CaptureRequest(BaseModel):
    filename: str
```

#### `QueryRequest`

```python
class QueryRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    user_input: str = Field(alias="userInput")
    context: List[Dict[str, Any]] = []
```

#### `EventRequest`

```python
class EventRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    source: str  # 'ocr' or 'asr'
    text: str
    metadata: Dict[str, Any] = {}
```

#### `ContextRequest`

```python
class ContextRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    count: int = 10
```

---

## OCR Module

Located in `backend/ocr.py`:

### `process_image()`

```python
def process_image(filename: str) -> None
```

- **Purpose**: Processes an image file for OCR
- **Input**: `filename` - path to image file
- **Output**: Currently outputs to `"temp_data/ocr_output.txt"`
- **Note**: This is a placeholder implementation

---

## ASR Module

Located in `backend/asr.py`:

### `process_audio()`

```python
def process_audio(filename: str) -> str
```

- **Purpose**: Processes audio file for speech recognition
- **Input**: `filename` - path to audio file
- **Returns**: Transcription result as string
- **Note**: Currently returns a mock transcription

---

## LLM Service

Located in `backend/llm.py`:

### Main LLM Service Class

```python
class LLMService:
    def __init__(self) -> None
        # Initialize LLM service for edge AI

    def load_model(self) -> None
        # Load edge-optimized model (ONNX or Transformers)

    def generate_response(self, user_input: str, context: List[Dict[str, Any]]) -> str
        # Generate response using edge AI model
        # Main function for LLM inference

    def get_model_info(self) -> Dict[str, Any]
        # Get information about the loaded model
```

### Key Methods

#### `generate_response()`

```python
def generate_response(self, user_input: str, context: List[Dict[str, Any]]) -> str
```

- **Purpose**: Main function for LLM inference
- **Input**:
  - `user_input`: User's query
  - `context`: List of context events
- **Returns**: Generated response string
- **Features**: Supports edge AI with NPU acceleration on Snapdragon X-Elite

#### `get_model_info()`

```python
def get_model_info(self) -> Dict[str, Any]
```

- **Purpose**: Get information about the loaded model
- **Returns**: Dictionary with model information including:
  - `model_name`: Name of the loaded model
  - `loaded`: Whether model is loaded
  - `edge_optimized`: Boolean indicating edge optimization
  - `platform`: Target platform (Snapdragon X-Elite)

### Global Instance

```python
llm_service = LLMService()  # Global instance ready for use
```

---

## Storage & Embedding System

Located in `backend/storage/`:

### Primary Storage Interface

From `backend/storage/interface.py` - these are the main functions exposed via `backend/storage/__init__.py`:

#### `store_raw_event()`

```python
def store_raw_event(
    session_id: str,
    source: str,
    ts: float,
    text: str,
    metadata: Dict[str, Any] = None
) -> str
```

- **Purpose**: Store raw event from OCR/ASR pipeline
- **Input**:
  - `session_id`: Unique session identifier
  - `source`: Source type ('ocr' or 'audio')
  - `ts`: Timestamp of the event
  - `text`: Raw text content
  - `metadata`: Additional metadata (optional)
- **Returns**: Unique event ID
- **Raises**: `ValueError` for invalid parameters

#### `process_session()`

```python
def process_session(session_id: str) -> List[str]
```

- **Purpose**: Process all raw events for a session and create embeddings
- **Input**: `session_id` - Session to process
- **Returns**: List of node IDs created for the session
- **Behavior**:
  1. Retrieves all raw events for the session
  2. Cleans and processes the data using the cleaning agent
  3. Creates embeddings for the processed data
  4. Stores results in both SQLite and FAISS
- **Raises**: `ValueError` for invalid session or `RuntimeError` for processing failures

#### `search_similar()`

```python
def search_similar(query: str, k: int = 5, filter: dict = None) -> List[Tuple[str, str]]
```

- **Purpose**: Search for similar content using semantic similarity
- **Input**:
  - `query`: Search query
  - `k`: Number of results to return (default: 5)
  - `filter`: Optional metadata filter dictionary
- **Returns**: List of tuples `(summary, full_data)` ordered by similarity
- **Raises**: `ValueError` for invalid parameters

#### `get_session_stats()`

```python
def get_session_stats(session_id: str) -> Dict[str, Any]
```

- **Purpose**: Get statistics for a specific session
- **Input**: `session_id` - Session to analyze
- **Returns**: Dictionary with session statistics:
  - `session_id`: Session identifier
  - `total_raw_events`: Total number of raw events
  - `ocr_events`: Number of OCR events
  - `audio_events`: Number of audio events
  - `is_processed`: Whether session has been processed
  - `exists`: Whether session exists

#### `get_system_stats()`

```python
def get_system_stats() -> Dict[str, Any]
```

- **Purpose**: Get overall system statistics
- **Returns**: Dictionary with system-wide statistics

#### `clear_all_data()`

```python
def clear_all_data() -> bool
```

- **Purpose**: Clear all stored data (use with caution)
- **Returns**: Success boolean

### Database Operations

From `backend/storage/db.py`:

#### `StorageDB` Class

```python
class StorageDB:
    def __init__(self, db_path: str = "storage.db") -> None

    def store_raw_event(self, session_id: str, source: str, ts: float, text: str, metadata: Dict[str, Any] = None) -> str
        # Store raw event in database

    def get_raw_events_by_session(self, session_id: str) -> List[Dict[str, Any]]
        # Get all raw events for a session

    def store_session_node(self, session_id: str, summary: str, full_data: str, embedding: bytes) -> str
        # Store processed session node with embedding

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]
        # Get node by ID

    def get_nodes_by_ids(self, node_ids: List[str]) -> List[Dict[str, Any]]
        # Get multiple nodes by IDs

    def get_all_nodes(self) -> List[Dict[str, Any]]
        # Get all nodes

    def session_exists(self, session_id: str) -> bool
        # Check if session exists

    def session_processed(self, session_id: str) -> bool
        # Check if session has been processed
```

### FAISS Vector Store

From `backend/storage/faiss_store.py`:

#### `FAISSStore` Class

```python
class FAISSStore:
    def __init__(self, index_dir: str = "faiss_index", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None

    def add_node(self, node_id: str, summary: str, full_data: str) -> bool
        # Add node to FAISS index

    def search_similar(self, query: str, k: int = 5, filter: dict = None) -> List[Tuple[str, str, float]]
        # Search for similar content with similarity scores

    def get_node_ids_by_similarity(self, query: str, k: int = 5, filter: dict = None) -> List[str]
        # Get node IDs by similarity

    def initialize_from_nodes(self, nodes: List[Dict[str, Any]]) -> bool
        # Initialize FAISS from existing nodes

    def get_stats(self) -> Dict[str, Any]
        # Get FAISS statistics

    def clear_index(self) -> bool
        # Clear FAISS index
```

---

## Model Management

### Edge Model Download

From `backend/download_model.py`:

#### `download_edge_model()`

```python
def download_edge_model() -> bool
```

- **Purpose**: Download small edge model (microsoft/DialoGPT-small)
- **Model**: 117M parameters - optimized for edge devices
- **Returns**: Success boolean
- **Download Time**: ~2-5 minutes

### Mistral Model Download

From `backend/download_mistral.py`:

#### `download_mistral_instruct()`

```python
def download_mistral_instruct() -> bool
```

- **Purpose**: Download Mistral Instruct model for edge AI
- **Model**: mistralai/Mistral-7B-Instruct-v0.2
- **Size**: ~14GB (quantized for edge devices)
- **Returns**: Success boolean
- **Download Time**: ~10-30 minutes

---

## Usage Examples

### Basic Workflow

1. **Store Events**:

```python
from storage import store_raw_event

# Store OCR result
event_id = store_raw_event("session123", "ocr", timestamp, "screenshot text")

# Store ASR result
event_id = store_raw_event("session123", "asr", timestamp, "audio transcription")
```

2. **Process Session**:

```python
from storage import process_session

# Process all events for a session
node_ids = process_session("session123")
```

3. **Search Similar Content**:

```python
from storage import search_similar

# Search for similar content
results = search_similar("what was I working on?", k=3)
```

4. **LLM Query**:

```python
from llm import llm_service

# Generate response
response = llm_service.generate_response("Summarize my work", context_events)
```

### API Integration

Frontend can call these endpoints:

```javascript
// Trigger OCR processing
fetch("/capture", {
  method: "POST",
  body: JSON.stringify({ filename: "screenshot.png" }),
  headers: { "Content-Type": "application/json" },
});

// Trigger ASR processing
fetch("/asr", { method: "POST" });

// Query LLM
fetch("/api/query", {
  method: "POST",
  body: JSON.stringify({
    sessionId: "session123",
    userInput: "What did I work on?",
    context: [],
  }),
  headers: { "Content-Type": "application/json" },
});

// Store event
fetch("/api/events", {
  method: "POST",
  body: JSON.stringify({
    sessionId: "session123",
    source: "ocr",
    text: "extracted text",
    metadata: { timestamp: "2025-01-12T10:00:00Z" },
  }),
  headers: { "Content-Type": "application/json" },
});
```

---

## Notes

- The system is designed for edge AI on Snapdragon X-Elite with NPU acceleration
- Storage system uses SQLite for persistence and FAISS for vector search
- LLM service supports both ONNX Runtime and Transformers backends
- All endpoints include CORS middleware for frontend integration
- Error handling is implemented throughout the API
