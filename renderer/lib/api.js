/**
 * API Contract for EdgeElite AI Assistant
 * 
 * This file defines the API interfaces that will be implemented by the backend team.
 * Currently using mock responses until the real endpoints are ready.
 */

const API_BASE_URL = 'http://localhost:8000'

// Mock data for development
const mockEvents = []
const mockContext = []

export const api = {
  // Event storage - Person 3 will implement
  storeEvent: async (sessionId, source, text, metadata = {}) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/events`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          sessionId, 
          source, 
          text, 
          metadata: {
            ...metadata,
            timestamp: new Date().toISOString()
          }
        })
      })
      
      if (!response.ok) {
        throw new Error(`Event storage failed: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.warn('Event storage failed, using mock:', error.message)
      
      // Mock response for development
      const mockEvent = {
        id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        sessionId,
        source,
        text,
        metadata,
        timestamp: new Date().toISOString()
      }
      
      mockEvents.push(mockEvent)
      console.log('Mock event stored:', mockEvent)
      
      return { 
        event_id: mockEvent.id, 
        status: 'stored',
        message: 'Event stored (mock mode)'
      }
    }
  },

  // Context retrieval - Person 3 will implement  
  getContext: async (sessionId, count = 10) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, count })
      })
      
      if (!response.ok) {
        throw new Error(`Context retrieval failed: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.warn('Context retrieval failed, using mock:', error.message)
      
      // Mock response for development
      const sessionEvents = mockEvents.filter(event => event.sessionId === sessionId)
      const recentEvents = sessionEvents.slice(-count)
      
      return {
        session_id: sessionId,
        context: recentEvents,
        count: recentEvents.length,
        message: 'Context retrieved (mock mode)'
      }
    }
  },

  // LLM query - You will implement
  queryLLM: async (sessionId, userInput, context = []) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          sessionId, 
          userInput, 
          context 
        })
      })
      
      if (!response.ok) {
        throw new Error(`LLM query failed: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.warn('LLM query failed, using mock:', error.message)
      
      // Mock response for development
      const contextSummary = context.length > 0 
        ? `Found ${context.length} context items` 
        : 'No context available'
      
      const mockResponse = `
AI Assistant Response (Mock Mode):
Based on the context (${contextSummary}), here's what I understand:

User Query: ${userInput}

Analysis: I can see you've been working with screenshots and audio recordings in session ${sessionId}. 
The system has captured ${context.length} events in your current session.

Recommendations:
1. Continue capturing relevant information
2. Use the summarize feature to get insights
3. Ask specific questions about your captured content

This is a mock response - real LLM integration coming soon!
      `.trim()
      
      return {
        response: mockResponse,
        session_id: sessionId,
        message: 'LLM response generated (mock mode)'
      }
    }
  },

  // Utility function to get session statistics
  getSessionStats: async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/stats`)
      
      if (!response.ok) {
        throw new Error(`Session stats failed: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.warn('Session stats failed, using mock:', error.message)
      
      // Mock response for development
      const sessionEvents = mockEvents.filter(event => event.sessionId === sessionId)
      const ocrEvents = sessionEvents.filter(event => event.source === 'ocr')
      const asrEvents = sessionEvents.filter(event => event.source === 'asr')
      
      return {
        session_id: sessionId,
        total_events: sessionEvents.length,
        ocr_events: ocrEvents.length,
        asr_events: asrEvents.length,
        message: 'Session stats retrieved (mock mode)'
      }
    }
  },

  // End session and trigger journal processing
  endSession: async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/session/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId })
      })
      
      if (!response.ok) {
        throw new Error(`Session end failed: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('Session end failed:', error)
      throw error
    }
  },

  // Poll for journal processing results
  pollJournal: async (sessionId) => {
    let status = "processing"
    let attempts = 0
    const maxAttempts = 40 // 60 seconds max
    
    while (status === "processing" && attempts < maxAttempts) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/journal`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId })
        })
        
        if (!response.ok) {
          throw new Error(`Journal poll failed: ${response.status}`)
        }
        
        const data = await response.json()
        status = data.status
        
        if (status === "done") {
          return data
        }
        
        // Wait 1.5 seconds before next poll
        await new Promise(resolve => setTimeout(resolve, 1500))
        attempts++
        
      } catch (error) {
        console.error('Journal polling error:', error)
        throw error
      }
    }
    
    throw new Error('Journal processing timeout')
  },

  // Context recall query
  contextRecall: async (sessionId, queryText) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/recall`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, queryText })
      })
      
      if (!response.ok) {
        throw new Error(`Recall query failed: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('Context recall failed:', error)
      throw error
    }
  },

  // Get all journal entries
  getJournalEntries: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/journal/entries`)
      
      if (!response.ok) {
        throw new Error(`Journal entries fetch failed: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('Journal entries fetch failed:', error)
      throw error
    }
  }
}

export default api 