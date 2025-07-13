# Phase 4: Demo Data Setup & Testing Guide

This guide explains how to set up demo data and test the EdgeElite Journal feature pipeline.

## 📋 Overview

Phase 4 involves:

1. **Demo Data Setup** - Create historical sessions for RAG testing
2. **Pipeline Testing** - Validate each component works correctly
3. **End-to-End Testing** - Test the complete journal workflow

## 🚀 Step 1: Set Up Demo Data

The demo data creates a historical session from "May 5th" with the walk-without-phone remedy that the RAG system can find.

### Run the Demo Data Seeder:

```bash
cd backend
python seed_demo_data.py
```

**What it does:**

- Creates a demo session `2025-05-05-demo` with realistic OCR and audio events
- Simulates a stressed user who finds relief through a 15-minute walk without phone
- Processes the session to create searchable embeddings
- Tests RAG retrieval to ensure the remedy can be found

**Expected Output:**

```
🌟 Creating May 5th Demo Session...
📷 Storing 8 OCR events...
🎤 Storing 7 Audio events...
🔄 Processing 3 demo sessions...
✅ Demo data seeding complete!
🔍 Testing RAG Retrieval with Demo Data...
🎯 FOUND THE WALK REMEDY!
```

## 🧪 Step 2: Test the Pipeline Components

Test individual components of the journal pipeline to ensure everything works:

### Run the Pipeline Test Suite:

```bash
cd backend
python test_journal_pipeline.py
```

**What it tests:**

1. **Data Ingestion** - OCR and Audio storage functions
2. **Session Processing** - Embedding generation pipeline
3. **RAG Retrieval** - Semantic search functionality

**Expected Output:**

```
🧪 TEST 1: Data Ingestion
✅ Stored 5 OCR events
✅ Stored 3 audio events
✅ Event count matches expected: 8

🧪 TEST 2: Session Processing
✅ Created 2 searchable chunks
✅ Session processed successfully

🧪 TEST 3: RAG Retrieval
🎯 Found content from test session!
✅ RAG retrieval working correctly

🧪 TEST SUMMARY
Tests passed: 3
Tests failed: 0
🎉 All tests passed!
```

## 🔍 Step 3: Test Demo Data Retrieval

Verify that the demo data can be found by semantic search:

```bash
cd backend
python -c "
from storage.interface import search_similar
results = search_similar('stressed headache need break', k=3)
for i, (summary, content) in enumerate(results, 1):
    print(f'{i}. {summary[:50]}...')
    print(f'   Content: {content[:100]}...')
    if 'walk without' in content.lower():
        print('   🎯 FOUND THE WALK REMEDY!')
"
```

## 🎯 Step 4: Test Complete Journal Workflow

Test the complete end-to-end journal workflow:

### 4.1 Start the Backend:

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4.2 Start the Frontend:

```bash
cd renderer
npm run dev
```

### 4.3 Navigate to Journal Page:

- Open Electron app
- Go to Journal page
- You should see the journal interface

### 4.4 Test the Demo Scenario:

1. **Start a new session** - Click "Start Session"
2. **Create stress context** - Say something like:
   ```
   "I have a huge headache today, my calendar is completely insane with back-to-back meetings"
   ```
3. **Take a screenshot** - Capture a busy calendar or task list
4. **End the session** - Click "End Session"
5. **Wait for processing** - The system will process the session and generate a journal entry
6. **Check the journal** - It should reference the May 5th walk remedy!

**Expected Journal Entry:**

```
You're feeling overwhelmed with a busy schedule and experiencing stress symptoms.
Try the same 15-minute walk without your phone that helped you feel calmer on May 5th.
This simple break can help clear your head and restore your focus.
```

## 📊 Step 5: Verify Database Content

Check that data is being stored correctly:

```bash
cd backend
python -c "
from storage.interface import get_system_stats, get_session_stats
stats = get_system_stats()
print(f'Total sessions: {stats[\"total_sessions_processed\"]}')
print(f'Total chunks: {stats[\"total_nodes\"]}')
print(f'Total events: {stats[\"total_raw_events\"]}')

# Check demo session
demo_stats = get_session_stats('2025-05-05-demo')
print(f'Demo session processed: {demo_stats[\"is_processed\"]}')
print(f'Demo session events: {demo_stats[\"total_raw_events\"]}')
"
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Import Errors**:

   ```
   cd backend
   pip install -r requirements.txt
   ```

2. **No Demo Data Found**:

   - Run `python seed_demo_data.py` first
   - Check if the session was processed correctly

3. **RAG Not Finding Remedy**:

   - Verify embeddings are created: check `backend/storage/faiss_index/`
   - Test different query phrases
   - Ensure the demo session contains the walk remedy text

4. **Frontend Not Connecting**:

   - Check that backend is running on port 8000
   - Verify API endpoints are accessible: `curl http://localhost:8000/api/journal`

5. **Journal Processing Timeout**:
   - Check backend logs for processing errors
   - Verify LLM service is working
   - Ensure session has sufficient data

### Debug Commands:

```bash
# Check database contents
cd backend
python -c "
from storage.db import StorageDB
db = StorageDB()
sessions = db.get_all_sessions()
print(f'Sessions in database: {len(sessions)}')
"

# Test search manually
python -c "
from storage.interface import search_similar
results = search_similar('walk without phone', k=5)
print(f'Search results: {len(results)}')
for summary, content in results:
    print(f'Found: {summary}')
"

# Check FAISS index
python -c "
from storage.faiss_store import FAISSStore
store = FAISSStore()
print(f'FAISS index size: {store.index.ntotal if store.index else 0}')
"
```

## ✅ Success Criteria

The pipeline is working correctly when:

- ✅ Demo data seeder runs without errors
- ✅ Test suite passes all tests
- ✅ RAG retrieval finds the walk remedy
- ✅ Journal generation produces relevant entries
- ✅ Frontend can start/end sessions
- ✅ Journal entries reference past experiences

## 🎉 Next Steps

Once Phase 4 is complete:

1. **Phase 5: Testing & Validation** - Final integration testing
2. **Demo Preparation** - Prepare live demonstration
3. **Documentation** - Update user guides and API docs

## 📝 Test Results Log

Keep track of your test results:

```
[ ] Demo data seeded successfully
[ ] Pipeline tests pass
[ ] RAG retrieval works
[ ] End-to-end workflow complete
[ ] Database content verified
[ ] Ready for Phase 5
```

---

**Need Help?** Check the logs in `backend/storage/` and ensure all dependencies are installed!
