import { useState, useRef } from 'react'
import './FileUpload.css'
import { BsUpload, BsFolder } from 'react-icons/bs'

function FileUpload({ backendUrl }) {
  const [file, setFile] = useState(null)
  const [files, setFiles] = useState([])  // For folder uploads
  const [uploadMode, setUploadMode] = useState('file')  // 'file' or 'folder'
  const [dragActive, setDragActive] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState({ current: 0, total: 0 })
  const [mode, setMode] = useState('auto')
  const [useAgentic, setUseAgentic] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const folderInputRef = useRef(null)

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

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (uploadMode === 'folder' || e.dataTransfer.files.length > 1) {
        // Multiple files dropped - treat as folder upload
        const fileList = Array.from(e.dataTransfer.files)
        setFiles(fileList)
        setFile(null)
        setUploadMode('folder')
      } else {
        setFile(e.dataTransfer.files[0])
        setFiles([])
      }
      setResult(null)
      setError(null)
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setFiles([])
      setUploadMode('file')
      setResult(null)
      setError(null)
    }
  }

  const handleFolderChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const fileList = Array.from(e.target.files)
      setFiles(fileList)
      setFile(null)
      setUploadMode('folder')
      setResult(null)
      setError(null)
    }
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleFolderButtonClick = () => {
    folderInputRef.current?.click()
  }

  const handleProcess = async () => {
    if (!file && files.length === 0) {
      setError('Please select a file or folder first')
      return
    }

    setProcessing(true)
    setError(null)
    setResult(null)
    setProcessingProgress({ current: 0, total: uploadMode === 'folder' ? files.length : 1 })

    try {
      if (uploadMode === 'folder' && files.length > 0) {
        // Generate a job ID for progress tracking
        const jobId = crypto.randomUUID()

        // Start listening to progress updates via SSE
        const eventSource = new EventSource(`${backendUrl}/api/progress/${jobId}`)
        eventSource.onmessage = (event) => {
          const progress = JSON.parse(event.data)
          setProcessingProgress({
            current: progress.current,
            total: progress.total,
            currentFile: progress.current_file
          })
          if (progress.status === 'completed' || progress.status === 'error') {
            eventSource.close()
          }
        }
        eventSource.onerror = () => {
          eventSource.close()
        }

        // Folder upload - send multiple files
        const formData = new FormData()
        files.forEach((f) => {
          // Preserve relative path if available
          const relativePath = f.webkitRelativePath || f.name
          formData.append('files', f)
          formData.append('paths', relativePath)
        })
        formData.append('mode', mode)
        formData.append('use_agentic', useAgentic)
        formData.append('job_id', jobId)

        const response = await fetch(`${backendUrl}/api/process-folder`, {
          method: 'POST',
          body: formData,
        })

        eventSource.close()

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || 'Processing failed')
        }

        const data = await response.json()
        setResult(data)
      } else {
        // Single file upload
        const formData = new FormData()
        formData.append('file', file)
        formData.append('mode', mode)
        formData.append('use_agentic', useAgentic)

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
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(false)
      setProcessingProgress({ current: 0, total: 0 })
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
    setFiles([])
    setUploadMode('file')
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    if (folderInputRef.current) {
      folderInputRef.current.value = ''
    }
  }

  return (
    <div className="file-upload">

      <div className="mode-selector">
        <label className="toggle-label">
          <span className="toggle-text">Processing Mode</span>
          <div className="toggle-container">
            <span className={`toggle-option ${!useAgentic ? 'active' : ''}`}>Standard</span>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={useAgentic}
                onChange={(e) => setUseAgentic(e.target.checked)}
                disabled={processing}
              />
              <span className="toggle-slider"></span>
            </label>
            <span className={`toggle-option ${useAgentic ? 'active' : ''}`}>Agentic</span>
          </div>
        </label>
        <p className="mode-description">
          {useAgentic
            ? 'Agentic mode uses multi-phase tool-calling for higher accuracy.'
            : 'Standard mode uses single-pass LLM processing.'}
        </p>
      </div>

      <div
        className={`drop-zone ${dragActive ? 'active' : ''} ${file || files.length > 0 ? 'has-file' : ''}`}
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
          accept=".png,.jpg,.jpeg,.csv,.txt,.dcm,.dicom,.pdf,.hea"
        />
        <input
          ref={folderInputRef}
          type="file"
          className="file-input"
          onChange={handleFolderChange}
          webkitdirectory=""
          directory=""
          multiple
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
        ) : files.length > 0 ? (
          <div className="folder-info">
            <div className="folder-header">
              <div className="file-icon"><BsFolder /></div>
              <div className="file-details">
                <p className="file-name">
                  {files[0].webkitRelativePath ? files[0].webkitRelativePath.split('/')[0] : 'Selected Files'}
                </p>
                <p className="file-size">
                  {files.length} files ({(files.reduce((sum, f) => sum + f.size, 0) / 1024).toFixed(2)} KB total)
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
            <div className="folder-file-list">
              {files.slice(0, 10).map((f, idx) => (
                <div key={idx} className="folder-file-item">
                  <span className="folder-file-name">{f.webkitRelativePath || f.name}</span>
                  <span className="folder-file-size">{(f.size / 1024).toFixed(1)} KB</span>
                </div>
              ))}
              {files.length > 10 && (
                <div className="folder-file-more">
                  ... and {files.length - 10} more files
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="drop-zone-content">
            <div className="upload-icon"><BsUpload /></div>
            <p className="drop-zone-text">
              Drag and drop your files here
            </p>
            <p className="drop-zone-subtext">or</p>
            <div className="browse-buttons">
              <button
                type="button"
                className="browse-button"
                onClick={handleButtonClick}
              >
                Browse Files
              </button>
              <button
                type="button"
                className="browse-button browse-folder"
                onClick={handleFolderButtonClick}
              >
                <BsFolder /> Browse Folder
              </button>
            </div>
            <p className="supported-formats">
              Supported: PNG, JPG, CSV, TXT, DICOM, PDF, HEA
            </p>
          </div>
        )}
      </div>

      {(file || files.length > 0) && !result && (
        <button
          className="process-button"
          onClick={handleProcess}
          disabled={processing}
        >
          {processing
            ? `Processing${processingProgress.total > 1 ? ` (${processingProgress.current}/${processingProgress.total})` : ''}...`
            : files.length > 0
              ? `Anonymize ${files.length} Files`
              : 'Anonymize File'}
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
            {result.files_processed
              ? `${result.files_processed} files processed successfully!`
              : 'File processed successfully!'}
            {result.files_failed > 0 && (
              <span className="warning-text"> ({result.files_failed} failed)</span>
            )}
          </div>

          {/* Single file result */}
          {result.filename_mapping && result.filename_mapping.phi_segments && result.filename_mapping.phi_segments.length > 0 && (
            <div className="filename-mapping">
              <h3>Filename Anonymization</h3>
              <div className="filename-comparison">
                <div className="filename-item">
                  <span className="filename-label">Original:</span>
                  <span className="filename-value">{result.filename_mapping.original_filename}</span>
                </div>
                <div className="filename-item">
                  <span className="filename-label">Anonymized:</span>
                  <span className="filename-value">{result.filename_mapping.anonymized_filename}</span>
                </div>
              </div>
              <div className="phi-segments">
                <h4>PHI Found in Filename:</h4>
                <ul>
                  {result.filename_mapping.phi_segments.map((seg, idx) => (
                    <li key={idx}>
                      <span className="phi-original">{seg.original_text}</span>
                      <span className="phi-arrow">→</span>
                      <span className="phi-anonymized">{seg.anonymized_text}</span>
                      <span className="phi-category">({seg.phi_category})</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Batch result - file mappings */}
          {result.file_results && result.file_results.length > 0 && (
            <div className="batch-results">
              <h3>Processed Files</h3>
              <div className="batch-file-list">
                {result.file_results.slice(0, 20).map((fileResult, idx) => (
                  <div key={idx} className={`batch-file-item ${fileResult.status === 'error' ? 'error' : ''}`}>
                    <span className="batch-file-original">{fileResult.original_path}</span>
                    <span className="batch-file-arrow">→</span>
                    <span className="batch-file-anonymized">
                      {fileResult.status === 'error' ? fileResult.error : fileResult.anonymized_filename}
                    </span>
                  </div>
                ))}
                {result.file_results.length > 20 && (
                  <div className="batch-file-more">
                    ... and {result.file_results.length - 20} more files
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="result-actions">
            <button
              className="download-button"
              onClick={handleDownload}
            >
              {result.is_batch ? 'Download ZIP Archive' : 'Download Anonymized File'}
            </button>
            <button
              className="process-another"
              onClick={handleReset}
            >
              Process Another {result.is_batch ? 'Folder' : 'File'}
            </button>
          </div>
        </div>
      )}

      {processing && (
        <div className="processing-overlay">
          <div className="spinner"></div>
          <p>Anonymizing your {files.length > 0 ? 'files' : 'file'}...</p>
          {processingProgress.total > 1 && (
            <>
              <p className="processing-progress">
                Processing file {processingProgress.current} of {processingProgress.total}
              </p>
              {processingProgress.currentFile && (
                <p className="processing-current-file">
                  {processingProgress.currentFile}
                </p>
              )}
            </>
          )}
          <p className="processing-subtext">
            This may take a few moments depending on {files.length > 0 ? 'the number of files and their sizes' : 'file size'}
          </p>
        </div>
      )}
    </div>
  )
}

export default FileUpload
