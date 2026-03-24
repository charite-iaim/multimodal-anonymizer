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

  const handleBackToConfig = () => {
    setIsConfigured(false)
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Multimodal Anonymizer</h1>
          <p className="subtitle">
            Securely deidentify personally identifiable information in your files using your own local multimodal LLM
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
            <FileUpload backendUrl={backendUrl} onBackToConfig={handleBackToConfig} />
          </div>
        )}

      </div>

      <footer className="footer">
        <p>
          This tool uses AI to detect and redact Personally Identifiable Information from medical records.
          Files are processed locally and not stored permanently.
        </p>
        <div className="footer-contact">
          <span>Developed by</span>
          <a href="mailto:anja.hirsch@charite.de">Anja Hirsch</a>
          <span>&</span>
          <a href="mailto:julian-gabriel.madrid@dhzc-charite.de">Julian Madrid</a>
        </div>
      </footer>
    </div>
  )
}

export default App
