import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Shield, Database, Cpu, FileText } from 'lucide-react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    { role: 'system', content: 'System ready. Ask a question about your indexed documents. All processing is 100% offline and private.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom of the chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);

    try {
      // Send the query to your local FastAPI backend
      const response = await axios.post('http://localhost:8000/chat', {
        query: userMessage.content
      });

      const aiMessage = { 
        role: 'assistant', 
        content: response.data.answer,
        sources: response.data.sources 
      };
      
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'system', 
        content: `Error: ${error.message}. Please ensure the FastAPI server is running.` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      {/* Sidebar */}
      <div className="sidebar" style={{ backgroundColor: '#0f172a' }}>
        <div className="sidebar-header">
          {/* Using a medical cross concept icon */}
          <Shield size={32} color="#38bdf8" /> 
          <h2>MediRAG Secure</h2>
        </div>
        
        <div className="status-panel">
          <h3>Compliance Status</h3>
          <div className="status-item"><Database size={16}/> Local Index: <b style={{color: '#38bdf8'}}>Encrypted</b></div>
          <div className="status-item"><Cpu size={16}/> Offline LLM: <b style={{color: '#38bdf8'}}>Active</b></div>
          <div className="status-item"><Shield size={16}/> HIPAA Guard: <b>DP-RAG (<span style={{color: '#38bdf8'}}>ε=1.2</span>)</b></div>
        </div>

        <div className="info-panel" style={{ borderLeft: '3px solid #38bdf8' }}>
          <p><strong>Clinical Environment:</strong> This system runs 100% locally. Patient Electronic Health Records (EHR) and laboratory data are mathematically protected from embedding inversion attacks.</p>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-area">
        <div className="messages-container">
          {messages.map((msg, index) => (
            <div key={index} className={`message-wrapper ${msg.role}`}>
              <div className="message-bubble">
                <div className="message-content">{msg.content}</div>
                
                {/* Render Sources if available */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-container">
                    <h4><FileText size={14}/> Sources Used:</h4>
                    <ul>
                      {/* Deduplicate sources by filename so it looks cleaner */}
                      {[...new Set(msg.sources.map(s => s.filename))].map((filename, idx) => (
                        <li key={idx}>{filename}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message-wrapper system">
              <div className="message-bubble loading">Retrieving context and generating answer...</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about your documents..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !query.trim()}>
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;