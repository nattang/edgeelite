# EdgeElite Module Structure Guide

This document shows exactly where each extension/module is located in the EdgeElite architecture.

## 🏗️ Overall Architecture

```
edgeelite/
├── src/app/                    # Next.js App Router (Frontend Pages)
│   ├── page.tsx               # Landing page
│   ├── journal/               # 🧠 Mental Health Journaling Module
│   ├── tutor/                 # 📚 Offline AI Tutor Module
│   └── accessibility/         # ♿ Accessibility Assistant Module
├── src/components/            # Reusable React Components
├── src/lib/                   # Utility Functions & Services
├── electron/                  # Desktop App (Electron)
├── python/                    # AI Services (Python)
├── models/                    # AI Model Files
└── data/                      # Local Data Storage
```

## 🧠 Mental Health Journaling Module

### **Frontend Location**: `src/app/journal/`
- **Main Page**: `src/app/journal/page.tsx`
- **Components**: `src/components/journal/`
- **Services**: `src/lib/journal/`

### **Features Implemented**:
- ✅ **Sentiment Analysis** - Real-time emotional analysis
- ✅ **Mood Tracking** - 1-10 scale mood ratings
- ✅ **Journal Entries** - Rich text journaling
- ✅ **Emotional Patterns** - Trend visualization
- ✅ **Privacy-First** - All data stays local

### **AI Integration**:
- **Model**: `python/models/inference.py` → `sentiment-analysis` model
- **Database**: `data/edgeelite.db` → `journal_entries` table
- **Vector Store**: `python/embeddings/faiss_manager.py` → `journal_embeddings` index

### **Data Flow**:
```
User Input → Sentiment Analysis → Database Storage → FAISS Indexing → Pattern Detection
```

---

## 📚 Offline AI Tutor Module

### **Frontend Location**: `src/app/tutor/`
- **Main Page**: `src/app/tutor/page.tsx`
- **Components**: `src/components/tutor/`
- **Services**: `src/lib/tutor/`

### **Features Implemented**:
- ✅ **Local LLM** - LLaMA-7B INT8 for Q&A
- ✅ **Topic Management** - Learning paths and progress
- ✅ **Adaptive Learning** - Personalized content
- ✅ **Confidence Scoring** - AI response quality
- ✅ **Offline Operation** - No internet required

### **AI Integration**:
- **Model**: `python/models/inference.py` → `llama-7b-int8` model
- **Database**: `data/edgeelite.db` → `learning_sessions` table
- **Vector Store**: `python/embeddings/faiss_manager.py` → `tutor_embeddings` index

### **Data Flow**:
```
Question → Local LLM → Answer Generation → Confidence Scoring → Session Storage
```

---

## ♿ Accessibility Assistant Module

### **Frontend Location**: `src/app/accessibility/`
- **Main Page**: `src/app/accessibility/page.tsx`
- **Components**: `src/components/accessibility/`
- **Services**: `src/lib/accessibility/`

### **Features Implemented**:
- ✅ **Object Detection** - Real-time environment analysis
- ✅ **OCR (Text Extraction)** - Image-to-text conversion
- ✅ **Voice Commands** - Speech-to-text control
- ✅ **Screen Reading** - Audio descriptions
- ✅ **Activity Logging** - Usage tracking

### **AI Integration**:
- **Models**: 
  - `python/models/inference.py` → `object-detection` model
  - `python/models/inference.py` → `troc-ocr` model
  - `python/models/inference.py` → `sentiment-analysis` (for voice)
- **Database**: `data/edgeelite.db` → `accessibility_logs` table

### **Data Flow**:
```
Camera/Image → Object Detection/OCR → Results → Voice Output → Activity Logging
```

---

## 🔧 Shared Infrastructure

### **Database Schema** (`data/edgeelite.db`)
```sql
-- Users table (shared across all modules)
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    settings TEXT
);

-- Journal entries (Mental Health module)
CREATE TABLE journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    sentiment_score REAL,
    sentiment_label TEXT,
    mood_rating INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Learning sessions (Tutor module)
CREATE TABLE learning_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Accessibility logs (Accessibility module)
CREATE TABLE accessibility_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    content TEXT,
    confidence_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### **FAISS Vector Indices** (`data/faiss/`)
```
data/faiss/
├── journal_embeddings.index      # Mental Health similarity search
├── journal_embeddings.metadata   # Journal entry metadata
├── tutor_embeddings.index        # Learning content retrieval
└── tutor_embeddings.metadata     # Tutor content metadata
```

### **AI Models** (`models/`)
```
models/
├── llama-7b-int8.onnx           # Text generation (Tutor)
├── minilm-embedding-int8.onnx   # Embeddings (All modules)
├── sentiment-analysis.onnx      # Sentiment analysis (Journal)
├── yolo-object-detection.onnx   # Object detection (Accessibility)
└── troc-ocr.onnx               # Text extraction (Accessibility)
```

---

## 🎯 Module Integration Points

### **Navigation** (`src/components/layout/`)
- **Sidebar Navigation** - Module switching
- **Header** - User info and settings
- **Breadcrumbs** - Navigation context

### **Shared Components** (`src/components/ui/`)
- **Glass Cards** - Consistent styling
- **Loading States** - AI processing indicators
- **Charts** - Data visualization
- **Forms** - Input components

### **Utility Services** (`src/lib/`)
- **Database Service** - SQLite operations
- **AI Service** - Model inference calls
- **Auth Service** - User authentication
- **Export Service** - Data export utilities

### **Electron Integration** (`electron/`)
- **Main Process** - Desktop app management
- **Preload Script** - Secure API exposure
- **IPC Handlers** - Module communication

---

## 🚀 Development Workflow

### **Adding New Features to Modules**:

1. **Frontend** (`src/app/[module]/`)
   ```bash
   # Add new page
   touch src/app/journal/analytics/page.tsx
   
   # Add component
   touch src/components/journal/MoodChart.tsx
   ```

2. **Backend** (`python/`)
   ```bash
   # Add new AI service
   touch python/services/mood_analyzer.py
   
   # Add new model
   touch python/models/mood_model.py
   ```

3. **Database** (`data/`)
   ```sql
   -- Add new table
   CREATE TABLE mood_patterns (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       user_id INTEGER NOT NULL,
       pattern_type TEXT NOT NULL,
       confidence REAL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

### **Module Communication**:
```typescript
// Example: Journal module calling AI service
import { analyzeSentiment } from '@/lib/ai/sentiment'

const result = await analyzeSentiment(journalEntry)
```

### **Data Sharing Between Modules**:
```typescript
// Example: Shared user context
import { useUser } from '@/lib/auth/user'

const { user, updateUser } = useUser()
```

---

## 📊 Module Performance Targets

### **Mental Health Journaling**:
- **Sentiment Analysis**: < 500ms
- **Mood Tracking**: < 100ms
- **Pattern Detection**: < 1 second

### **Offline AI Tutor**:
- **Question Answering**: < 2 seconds
- **Content Generation**: < 3 seconds
- **Learning Progress**: < 200ms

### **Accessibility Assistant**:
- **Object Detection**: < 300ms
- **OCR Processing**: < 1 second
- **Voice Recognition**: < 500ms

---

## 🔒 Security & Privacy

### **Data Isolation**:
- **User-specific data** - All tables have `user_id` foreign keys
- **Module separation** - Each module has its own data tables
- **Local storage** - No cloud dependencies

### **Privacy Features**:
- **Encryption** - Sensitive data encrypted at rest
- **No telemetry** - Zero data transmission
- **User control** - Full data export/deletion

---

## 🎨 UI/UX Consistency

### **Design System**:
- **Glassmorphism** - Consistent glass effects
- **Color Coding** - Module-specific colors
- **Typography** - Unified font system
- **Animations** - Smooth transitions

### **Responsive Design**:
- **Mobile-first** - Works on all screen sizes
- **Touch-friendly** - Accessibility considerations
- **Keyboard navigation** - Full keyboard support

---

## 📈 Scalability Considerations

### **Horizontal Scaling**:
- **Module independence** - Each module can scale separately
- **Database optimization** - Proper indexing for performance
- **Caching strategy** - Multi-level caching

### **Future Extensions**:
- **Plugin system** - Easy to add new modules
- **API abstraction** - Clean interfaces for new features
- **Configuration-driven** - Easy customization

---

**This structure ensures clean separation of concerns while maintaining tight integration between modules for a seamless user experience.** 🚀 