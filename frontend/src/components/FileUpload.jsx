import { useState, useRef } from 'react'
import './FileUpload.css'
import { BsUpload } from 'react-icons/bs'

function FileUpload({ backendUrl }) {
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [mode, setMode] = useState('auto')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
      setResult(null)
      setError(null)
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
      setError(null)
    }
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleProcess = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setProcessing(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('mode', mode)

      const response = await fetch(`${backendUrl}/api/process`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Processing failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(false)
    }
  }

  const handleDownload = async () => {
    if (!result?.download_url) return

    try {
      const response = await fetch(`${backendUrl}${result.download_url}`)
      if (!response.ok) throw new Error('Download failed')

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = result.download_url.split('/').pop()
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)

      // Cleanup after successful download
      await fetch(`${backendUrl}/api/cleanup/${result.job_id}`, {
        method: 'DELETE'
      })
    } catch (err) {
      setError('Failed to download file: ' + err.message)
    }
  }

  const handleReset = async () => {
    // Check if backend config is still valid before resetting
    try {
      const response = await fetch(`${backendUrl}/api/config/status`)
      const data = await response.json()

      if (!data.configured) {
        // Config was lost (backend restart?), reload page to show config form
        window.location.reload()
        return
      }
    } catch (err) {
      console.error('Failed to check config status:', err)
      // On error, reload to be safe
      window.location.reload()
      return
    }

    // Config is still valid, just reset the file upload state
    setFile(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="file-upload">

      <div
        className={`drop-zone ${dragActive ? 'active' : ''} ${file ? 'has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="file-input"
          onChange={handleFileChange}
          accept=".png,.jpg,.jpeg,.csv,.txt,.dcm,.dicom,.pdf"
        />

        {file ? (
          <div className="file-info">
            <div className="file-icon">📄</div>
            <div className="file-details">
              <p className="file-name">{file.name}</p>
              <p className="file-size">
                {(file.size / 1024).toFixed(2)} KB
              </p>
            </div>
            <button
              className="remove-file"
              onClick={handleReset}
              disabled={processing}
            >
              ✕
            </button>
          </div>
        ) : (
          <div className="drop-zone-content">
            <div className="upload-icon"><BsUpload /></div>
            <p className="drop-zone-text">
              Drag and drop your file here
            </p>
            <p className="drop-zone-subtext">or</p>
            <button
              type="button"
              className="browse-button"
              onClick={handleButtonClick}
            >
              Browse Files
            </button>
            <p className="supported-formats">
              Supported: PNG, JPG, CSV, TXT, DICOM, PDF
            </p>
          </div>
        )}
      </div>

      {file && !result && (
        <button
          className="process-button"
          onClick={handleProcess}
          disabled={processing}
        >
          {processing ? 'Processing...' : 'Anonymize File'}
        </button>
      )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="result-section">
          <div className="success-message">
            File processed successfully!
          </div>
          <div className="result-actions">
            <button
              className="download-button"
              onClick={handleDownload}
            >
              Download Anonymized File
            </button>
            <button
              className="process-another"
              onClick={handleReset}
            >
              Process Another File
            </button>
          </div>
        </div>
      )}

      {processing && (
        <div className="processing-overlay">
          <div className="spinner"></div>
          <p>Anonymizing your file...</p>
          <p className="processing-subtext">
            This may take a few moments depending on file size
          </p>
        </div>
      )}
    </div>
  )
}

export default FileUpload
