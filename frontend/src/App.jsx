import { useState } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import ConfigForm from './components/ConfigForm'

function App() {
  const [isConfigured, setIsConfigured] = useState(false)
  const [backendUrl, setBackendUrl] = useState(() => {
    return localStorage.getItem('backendUrl') || 'http://localhost:8000'
  })

  const handleBackendUrlChange = (url) => {
    setBackendUrl(url)
    localStorage.setItem('backendUrl', url)
  }

  const handleConfigured = () => {
    setIsConfigured(true)
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Multimodal Anonymization Tool</h1>
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
            <h2>LLM Configuration</h2>
            <ConfigForm backendUrl={backendUrl} onConfigured={handleConfigured} />
          </div>
        ) : (
          <div className="upload-section">
            <FileUpload backendUrl={backendUrl} />
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
