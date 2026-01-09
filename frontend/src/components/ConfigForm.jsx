import { useState } from 'react'
import './ConfigForm.css'

function ConfigForm({ backendUrl, onConfigured }) {
  const [mode, setMode] = useState('dev') // 'dev' or 'custom'
  const [customUrl, setCustomUrl] = useState(() => {
    return localStorage.getItem('customLlmUrl') || ''
  })
  const [customApiKey, setCustomApiKey] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(false)

  const handleDevMode = async () => {
    setLoading(true)
    setError(null)
    setSuccess(false)

    try {
      const response = await fetch(`${backendUrl}/api/config/dev`, {
        method: 'POST',
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to load dev configuration')
      }

      setSuccess(true)
      setTimeout(() => {
        onConfigured()
      }, 1000)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleCustomMode = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(false)

    try {
      const response = await fetch(`${backendUrl}/api/config/custom`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          llm_url: customUrl,
          api_key: customApiKey || null
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to configure custom LLM')
      }

      // Save URL to localStorage (not the API key)
      localStorage.setItem('customLlmUrl', customUrl)

      setSuccess(true)
      setTimeout(() => {
        onConfigured()
      }, 1000)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="config-form">
      <div className="mode-selector">
        <label>Select Configuration Mode:</label>
        <div className="mode-buttons">
          <button
            type="button"
            className={`mode-button ${mode === 'dev' ? 'active' : ''}`}
            onClick={() => setMode('dev')}
            disabled={loading}
          >
            Dev Mode
          </button>
          <button
            type="button"
            className={`mode-button ${mode === 'custom' ? 'active' : ''}`}
            onClick={() => setMode('custom')}
            disabled={loading}
          >
            Custom LLM
          </button>
        </div>
      </div>

      {mode === 'dev' ? (
        <div className="dev-mode-section">
          <div className="info-box">
            <h3>Development Mode</h3>
            <p>
              Uses credentials from your backend's environment variables.
              Make sure you have set:
            </p>
            <ul>
              <li>AZURE_OPENAI_ENDPOINT</li>
              <li>AZURE_OPENAI_API_KEY</li>
              <li>AZURE_OPENAI_DEPLOYMENT_NAME</li>
            </ul>
          </div>

          <button
            type="button"
            className="submit-button"
            onClick={handleDevMode}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Use Dev Configuration'}
          </button>
        </div>
      ) : (
        <form onSubmit={handleCustomMode} className="custom-mode-section">
          <div className="info-box">
            <h3>Custom LLM</h3>
            <p>
              Use your own local LLM or OpenAI-compatible endpoint.
            </p>
          </div>

          <div className="form-group">
            <label htmlFor="custom_url">
              LLM Endpoint URL
              <span className="required">*</span>
            </label>
            <input
              type="url"
              id="custom_url"
              value={customUrl}
              onChange={(e) => setCustomUrl(e.target.value)}
              placeholder="http://localhost:11434/v1"
              required
            />
            <small>Your local LLM or OpenAI-compatible API endpoint</small>
          </div>

          <div className="form-group">
            <label htmlFor="custom_api_key">
              API Key (Optional)
            </label>
            <input
              type="password"
              id="custom_api_key"
              value={customApiKey}
              onChange={(e) => setCustomApiKey(e.target.value)}
              placeholder="Leave empty if not required"
            />
            <small>API key if your LLM requires authentication</small>
          </div>

          <button
            type="submit"
            className="submit-button"
            disabled={loading}
          >
            {loading ? 'Connecting...' : 'Connect to LLM'}
          </button>
        </form>
      )}

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {success && (
        <div className="success-message">
          Configuration saved successfully!
        </div>
      )}
    </div>
  )
}

export default ConfigForm
