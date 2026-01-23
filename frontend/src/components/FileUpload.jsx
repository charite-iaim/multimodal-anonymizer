import { useState, useRef, useEffect } from 'react'
import './FileUpload.css'
import { BsUpload, BsFolder, BsCameraVideo, BsExclamationTriangle } from 'react-icons/bs'
import PromptSettings from './PromptSettings'

// Files to ignore when uploading folders (system files, hidden files, etc.)
const IGNORED_FILES = ['.DS_Store', 'Thumbs.db', '.gitkeep', '.gitignore']
const shouldIgnoreFile = (file) => {
  const fileName = file.name || ''
  // Ignore files starting with . (hidden files) or in the ignored list
  return fileName.startsWith('.') || IGNORED_FILES.includes(fileName)
}

// Video file extensions
const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
const isVideoFile = (file) => {
  const ext = '.' + file.name.split('.').pop().toLowerCase()
  return VIDEO_EXTENSIONS.includes(ext)
}

// CT/MRI file extensions
const CT_MRI_EXTENSIONS = ['.nii', '.nii.gz', '.nrrd', '.mha', '.mhd']
const isCTMRIFile = (file) => {
  const name = file.name.toLowerCase()
  return CT_MRI_EXTENSIONS.some(ext => name.endsWith(ext))
}

function FileUpload({ backendUrl }) {
  const [file, setFile] = useState(null)
  const [files, setFiles] = useState([])  // For folder uploads
  const [uploadMode, setUploadMode] = useState('file')  // 'file' or 'folder'
  const [dragActive, setDragActive] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState({ current: 0, total: 0 })
  const [promptSettingsOpen, setPromptSettingsOpen] = useState(false)
  const [videoSettingsOpen, setVideoSettingsOpen] = useState(false)
  const [processAllFrames, setProcessAllFrames] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [features, setFeatures] = useState(null)  // Feature availability
  const fileInputRef = useRef(null)
  const folderInputRef = useRef(null)

  // Check if any selected files are videos
  const hasVideoFiles = file ? isVideoFile(file) : files.some(f => isVideoFile(f))

  // Check if any selected files are CT/MRI
  const hasCTMRIFiles = file ? isCTMRIFile(file) : files.some(f => isCTMRIFile(f))

  // Fetch features and video settings on mount and when backend URL changes
  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/features`)
        if (response.ok) {
          const data = await response.json()
          setFeatures(data)
        }
      } catch (err) {
        console.error('Failed to fetch features:', err)
      }
    }
    fetchFeatures()
  }, [backendUrl])

  useEffect(() => {
    const fetchVideoSettings = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/video-settings`)
        if (response.ok) {
          const data = await response.json()
          setProcessAllFrames(data.process_all_frames)
        }
      } catch (err) {
        console.error('Failed to fetch video settings:', err)
      }
    }
    fetchVideoSettings()
  }, [backendUrl])

  // Update video settings when changed
  const handleVideoSettingChange = async (newValue) => {
    setProcessAllFrames(newValue)
    try {
      await fetch(`${backendUrl}/api/video-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ process_all_frames: newValue })
      })
    } catch (err) {
      console.error('Failed to update video settings:', err)
    }
  }

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
        const fileList = Array.from(e.dataTransfer.files).filter(f => !shouldIgnoreFile(f))
        if (fileList.length > 0) {
          setFiles(fileList)
          setFile(null)
          setUploadMode('folder')
        }
      } else {
        const droppedFile = e.dataTransfer.files[0]
        if (!shouldIgnoreFile(droppedFile)) {
          setFile(droppedFile)
          setFiles([])
        }
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
      const fileList = Array.from(e.target.files).filter(f => !shouldIgnoreFile(f))
      if (fileList.length > 0) {
        setFiles(fileList)
        setFile(null)
        setUploadMode('folder')
        setResult(null)
        setError(null)
      }
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

      {/* Prompt customization */}
      <PromptSettings
        backendUrl={backendUrl}
        isOpen={promptSettingsOpen}
        onToggle={() => setPromptSettingsOpen(!promptSettingsOpen)}
      />

      {/* CT/MRI processing warning - show when CT/MRI files are selected but feature is unavailable */}
      {hasCTMRIFiles && features && !features.ct_mri_processing?.available && (
        <div className="feature-warning">
          <div className="feature-warning-header">
            <BsExclamationTriangle className="warning-icon" />
            <span className="feature-warning-title">CT/MRI Processing Not Available</span>
          </div>
          <div className="feature-warning-content">
            <p>
              The selected CT/MRI files ({files.filter(f => isCTMRIFile(f)).length || 1} file{(files.filter(f => isCTMRIFile(f)).length || 1) > 1 ? 's' : ''}) require additional setup to process.
            </p>
            <p className="feature-warning-details">
              CT/MRI processing requires a separate Python 3.11 environment with the mede library installed.
              See <code>docs/MEDE_SETUP.md</code> for setup instructions.
            </p>
          </div>
        </div>
      )}

      {/* Video processing settings - show when video files are selected */}
      {hasVideoFiles && (
        <div className="video-settings">
          <div className="video-settings-header">
            <BsCameraVideo className="video-icon" />
            <span className="video-settings-title">Video Processing Settings</span>
          </div>
          <div className="video-settings-content">
            <label className="video-checkbox-label">
              <input
                type="checkbox"
                className="video-checkbox"
                checked={processAllFrames}
                onChange={(e) => handleVideoSettingChange(e.target.checked)}
                disabled={processing}
              />
              <div className="video-checkbox-info">
                <span className="video-checkbox-text">Analyze every frame</span>
                <span className="video-checkbox-description">
                  PHI detection runs on each frame individually (much slower, catches frame-specific PHI); otherwise, only the first frame is analyzed and its results are applied to all frames.
                </span>
              </div>
            </label>
          </div>
        </div>
      )}

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
          accept=".png,.jpg,.jpeg,.csv,.txt,.dcm,.dicom,.pdf,.hea,.docx,.xlsx,.xls,.mp4,.avi,.mov,.mkv,.wav,.mp3,.nii,.nii.gz,.nrrd,.mha,.mhd"
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
              Supported: PNG, JPG, CSV, TXT, DICOM, PDF, HEA, DOCX, Excel, MP4, AVI, MOV, WAV, MP3
              {features?.ct_mri_processing?.available && ', NIfTI, NRRD, MHA'}
            </p>
            {features?.ct_mri_processing?.available && (
              <p className="format-hint">
                💡 For 3D CT/MRI scans: name folders with suffix <code>_extended_3d_image</code>
              </p>
            )}
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
            ? `Processing${processingProgress.total > 1 && processingProgress.current > 0 ? ` (${processingProgress.current}/${processingProgress.total})` : ''}...`
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
                {processingProgress.current > 0
                  ? `File ${processingProgress.current} of ${processingProgress.total}`
                  : `Preparing ${processingProgress.total} files...`}
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
