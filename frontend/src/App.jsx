import { useState, useEffect } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'

function App() {
  const [isConfigured, setIsConfigured] = useState(false)
  const [loading, setLoading] = useState(true)
  const [configError, setConfigError] = useState(null)
  const [backendUrl, setBackendUrl] = useState(() => {
    return localStorage.getItem('backendUrl') || 'http://localhost:8000'
  })

  useEffect(() => {
    checkAndAutoConfigureAzure()
  }, [backendUrl])

  const checkAndAutoConfigureAzure = async () => {
    setConfigError(null)
    try {
      const response = await fetch(`${backendUrl}/api/config/status`)
      const data = await response.json()

      if (data.configured) {
        setIsConfigured(true)
        setLoading(false)
      } else {
        // Auto-configure with Azure (dev mode)
        const configResponse = await fetch(`${backendUrl}/api/config/dev`, {
          method: 'POST',
        })

        if (configResponse.ok) {
          setIsConfigured(true)
        } else {
          const errorData = await configResponse.json()
          setConfigError(errorData.detail || 'Failed to auto-configure Azure')
          setIsConfigured(false)
        }
        setLoading(false)
      }
    } catch (error) {
      console.error('Failed to check/configure:', error)
      setConfigError('Failed to connect to backend')
      setIsConfigured(false)
      setLoading(false)
    }
  }

  const handleBackendUrlChange = (url) => {
    setBackendUrl(url)
    localStorage.setItem('backendUrl', url)
    setLoading(true)
    checkAndAutoConfigureAzure()
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
            <h2>Azure Configuration</h2>
            {configError ? (
              <div className="error-message">
                <p>{configError}</p>
                <p className="info-text">
                  Make sure your backend has the following environment variables set:
                </p>
                <ul>
                  <li>AZURE_OPENAI_ENDPOINT</li>
                  <li>AZURE_OPENAI_API_KEY</li>
                  <li>AZURE_OPENAI_DEPLOYMENT_NAME</li>
                </ul>
                <button
                  className="submit-button"
                  onClick={() => checkAndAutoConfigureAzure()}
                >
                  Retry
                </button>
              </div>
            ) : (
              <p className="info-text">Configuring Azure OpenAI...</p>
            )}
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
