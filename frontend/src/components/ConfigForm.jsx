import { useState } from 'react'
import './ConfigForm.css'

function ConfigForm({ backendUrl, onConfigured }) {
  const [customUrl, setCustomUrl] = useState(() => {
    return localStorage.getItem('customLlmUrl') || ''
  })
  const [customModel, setCustomModel] = useState(() => {
    return localStorage.getItem('customLlmModel') || 'llama3.2'
  })
  const [customApiKey, setCustomApiKey] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(false)

    // Auto-prepend http:// if the user omitted the protocol
    let url = customUrl.trim()
    if (url && !url.startsWith('http://') && !url.startsWith('https://')) {
      url = `http://${url}`
      setCustomUrl(url)
    }

    try {
      let response
      try {
        response = await fetch(`${backendUrl}/api/config/custom`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            llm_url: url,
            model_name: customModel,
            api_key: customApiKey || null
          })
        })
      } catch (fetchErr) {
        throw new Error(
          `Cannot reach the backend at ${backendUrl}. Make sure the backend server is running.`
        )
      }

      if (!response.ok) {
        let detail = 'Failed to connect to LLM'
        try {
          const errorData = await response.json()
          detail = errorData.detail || detail
        } catch {
          // response body wasn't JSON
        }
        throw new Error(detail)
      }

      // Save URL and model to localStorage (not the API key)
      localStorage.setItem('customLlmUrl', customUrl)
      localStorage.setItem('customLlmModel', customModel)

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
      <form onSubmit={handleSubmit} className="custom-mode-section">
        <div className="info-box">
          <h3>Local LLM</h3>
          <p>
            Connect to any local LLM server with an OpenAI-compatible API:
          </p>
          <ul>
            <li><strong>Ollama</strong> - ollama.com</li>
            <li><strong>LM Studio</strong> - lmstudio.ai</li>
            <li><strong>vLLM</strong> - vllm.ai</li>
            <li><strong>LocalAI</strong> - localai.io</li>
          </ul>
        </div>

        <div className="form-group">
          <label htmlFor="custom_url">
            LLM Endpoint URL
            <span className="required">*</span>
          </label>
          <input
            type="text"
            id="custom_url"
            value={customUrl}
            onChange={(e) => setCustomUrl(e.target.value)}
            placeholder="http://localhost:11434/v1"
            required
          />
          <small>
            Common endpoints: Ollama (localhost:11434/v1), LM Studio (localhost:1234/v1), vLLM (localhost:8000/v1)
          </small>
        </div>

        <div className="form-group">
          <label htmlFor="custom_model">
            Model Name
            <span className="required">*</span>
          </label>
          <input
            type="text"
            id="custom_model"
            value={customModel}
            onChange={(e) => setCustomModel(e.target.value)}
            placeholder="llama3.2"
            required
          />
          <small>
            Model name as shown in your LLM server (e.g., llama3.2, mistral, qwen2.5:14b)
          </small>
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
          <small>Most local LLM servers don't require an API key</small>
        </div>

        <button
          type="submit"
          className="submit-button"
          disabled={loading}
        >
          {loading ? 'Testing connection...' : 'Connect to LLM'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {success && (
        <div className="success-message">
          Connected successfully!
        </div>
      )}
    </div>
  )
}

export default ConfigForm
