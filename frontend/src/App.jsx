import { useState, useEffect } from 'react'
import './App.css'
import ConfigForm from './components/ConfigForm'
import FileUpload from './components/FileUpload'

function App() {
  const [isConfigured, setIsConfigured] = useState(false)
  const [loading, setLoading] = useState(true)
  const [backendUrl, setBackendUrl] = useState(() => {
    return localStorage.getItem('backendUrl') || 'http://localhost:8000'
  })

  useEffect(() => {
    checkConfig()
  }, [backendUrl])

  const checkConfig = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/config/status`)
      const data = await response.json()
      setIsConfigured(data.configured)
    } catch (error) {
      console.error('Failed to check config status:', error)
      setIsConfigured(false)
    } finally {
      setLoading(false)
    }
  }

  const handleConfigured = () => {
    setIsConfigured(true)
  }

  const handleBackendUrlChange = (url) => {
    setBackendUrl(url)
    localStorage.setItem('backendUrl', url)
    setLoading(true)
    checkConfig()
  }

  if (loading) {
    return (
      <div className="app">
        <div className="container">
          <div className="loading">Connecting to backend...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>PII Anonymization Tool</h1>
          <p className="subtitle">
            Securely anonymize Personally Identifiable Information in your files using your own local LLM
          </p>
        </header>

        <div className="backend-url-section">
          <label htmlFor="backend-url">Backend URL:</label>
          <input
            id="backend-url"
            type="text"
            value={backendUrl}
            onChange={(e) => handleBackendUrlChange(e.target.value)}
            placeholder="http://localhost:8000"
          />
        </div>

        {!isConfigured ? (
          <div className="config-section">
            <h2>Configure LLM Endpoint</h2>
            <p className="info-text">
              Before processing files, configure your Azure OpenAI endpoint.
            </p>
            <ConfigForm
              backendUrl={backendUrl}
              onConfigured={handleConfigured}
            />
          </div>
        ) : (
          <div className="upload-section">
            <FileUpload backendUrl={backendUrl} />
            <button
              className="reconfigure-button"
              onClick={() => setIsConfigured(false)}
            >
              Reconfigure LLM Settings
            </button>
          </div>
        )}

        <footer className="footer">
          <p>
            This tool uses AI to detect and redact Personally Identifiable Information from medical records.
            Files are processed locally and not stored permanently.
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App
