import { useState, useRef, useEffect } from 'react'
import './FileUpload.css'
import { BsUpload, BsFolder, BsCameraVideo, BsExclamationTriangle, BsFileEarmark, BsX, BsChevronUp, BsChevronDown, BsArrowLeft } from 'react-icons/bs'
import PromptSettings from './PromptSettings'

const FUN_MESSAGES = [
  "Time to grab a coffee ☕",
  "Still watching? 👀",
  "Anonymization in progress... almost there!",
  "Good things come to those who wait..."
]

// Files to ignore when uploading folders (system files, hidden files, etc.)
const IGNORED_FILES = ['.DS_Store', 'Thumbs.db', '.gitkeep', '.gitignore']
const shouldIgnoreFile = (file) => {
  const fileName = file.name || ''
  // Ignore files starting with . (hidden files) or in the ignored list
  return fileName.startsWith('.') || IGNORED_FILES.includes(fileName)
}

// Video file extensions
const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
// DICOM file extensions
const DICOM_EXTENSIONS = ['.dcm', '.dicom']

const formatFileSize = (bytes) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

const isVideoExtension = (file) => {
  const ext = '.' + file.name.split('.').pop().toLowerCase()
  return VIDEO_EXTENSIONS.includes(ext)
}

const isDicomExtension = (file) => {
  const ext = '.' + file.name.split('.').pop().toLowerCase()
  return DICOM_EXTENSIONS.includes(ext)
}

function FileUpload({ backendUrl, onBackToConfig }) {
  const [files, setFiles] = useState([])
  const [dragActive, setDragActive] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState({ current: 0, total: 0 })
  const [promptSettingsOpen, setPromptSettingsOpen] = useState(false)
  const [videoSettingsOpen, setVideoSettingsOpen] = useState(false)
  const [processAllFrames, setProcessAllFrames] = useState(false)
  const [saveMappingFiles, setSaveMappingFiles] = useState(true)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [fileInfo, setFileInfo] = useState(null)  // Info about the selected file (e.g., DICOM video detection)
  const [checkingFileInfo, setCheckingFileInfo] = useState(false)
  const [funMessage, setFunMessage] = useState('')
  const [fileListExpanded, setFileListExpanded] = useState(true)
  const fileInputRef = useRef(null)
  const folderInputRef = useRef(null)

  // Check if any selected files are videos (either by extension or by DICOM video detection)
  const hasVideoFiles = (() => {
    if (files.some(f => isVideoExtension(f))) return true
    // Check if single DICOM file was detected as video
    if (fileInfo?.is_video && fileInfo?.supports_frame_by_frame) return true
    return false
  })()

  // Rotate messages while processing
  useEffect(() => {
    if (!processing) {
      setFunMessage('')
      return
    }
    const pick = () => FUN_MESSAGES[Math.floor(Math.random() * FUN_MESSAGES.length)]
    setFunMessage(pick())
    const interval = setInterval(() => setFunMessage(pick()), 60_000)
    return () => clearInterval(interval)
  }, [processing])

  // Fetch video settings on mount and when backend URL changes
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

  useEffect(() => {
    const fetchMappingFilesSettings = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/mapping-files-settings`)
        if (response.ok) {
          const data = await response.json()
          setSaveMappingFiles(data.save_mapping_files)
        }
      } catch (err) {
        console.error('Failed to fetch mapping files settings:', err)
      }
    }
    fetchMappingFilesSettings()
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

  // Update mapping files settings when changed
  const handleMappingFilesSettingChange = async (newValue) => {
    setSaveMappingFiles(newValue)
    try {
      await fetch(`${backendUrl}/api/mapping-files-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ save_mapping_files: newValue })
      })
    } catch (err) {
      console.error('Failed to update mapping files settings:', err)
    }
  }

  // Check if a file is a DICOM video (multi-frame) using the backend API
  const checkFileInfo = async (fileToCheck) => {
    // Only check DICOM files - other files don't need this check
    if (!isDicomExtension(fileToCheck)) {
      setFileInfo(null)
      return
    }

    setCheckingFileInfo(true)
    try {
      const formData = new FormData()
      formData.append('file', fileToCheck)

      const response = await fetch(`${backendUrl}/api/file-info`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const info = await response.json()
        setFileInfo(info)
        if (info.is_video && info.supports_frame_by_frame) {
          console.log(`DICOM video detected: ${info.frame_count} frames`)
        }
      } else {
        setFileInfo(null)
      }
    } catch (err) {
      console.error('Failed to check file info:', err)
      setFileInfo(null)
    } finally {
      setCheckingFileInfo(false)
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

  const handleDrop = async (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const items = e.dataTransfer.items
    if (!items || items.length === 0) return

    // Check if any dropped item is a directory
    const entries = []
    for (let i = 0; i < items.length; i++) {
      const entry = items[i].webkitGetAsEntry?.() || items[i].getAsEntry?.()
      if (entry) entries.push(entry)
    }

    const hasDirectory = entries.some(entry => entry.isDirectory)
    const newFiles = []

    if (hasDirectory) {
      // Recursively read all files from dropped directories (and include loose files)
      const readEntry = (entry, path) => {
        return new Promise((resolve) => {
          if (entry.isFile) {
            entry.file((file) => {
              if (!shouldIgnoreFile(file)) {
                file._relativePath = path + file.name
                newFiles.push(file)
              }
              resolve()
            }, () => resolve())
          } else if (entry.isDirectory) {
            const reader = entry.createReader()
            const readBatch = () => {
              reader.readEntries(async (dirEntries) => {
                if (dirEntries.length === 0) {
                  resolve()
                  return
                }
                await Promise.all(
                  dirEntries.map(child => readEntry(child, path + entry.name + '/'))
                )
                readBatch()
              }, () => resolve())
            }
            readBatch()
          } else {
            resolve()
          }
        })
      }

      await Promise.all(entries.map(entry => readEntry(entry, '')))
    } else if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      Array.from(e.dataTransfer.files).filter(f => !shouldIgnoreFile(f)).forEach(f => newFiles.push(f))
    }

    if (newFiles.length > 0) {
      // Append to existing files
      setFiles(prev => [...prev, ...newFiles])
    }
    setResult(null)
    setError(null)
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files).filter(f => !shouldIgnoreFile(f))
      if (newFiles.length > 0) {
        // Append to existing files
        setFiles(prev => [...prev, ...newFiles])

        setResult(null)
        setError(null)
        // Check if any new DICOM file is a video
        if (newFiles.length === 1 && isDicomExtension(newFiles[0])) {
          checkFileInfo(newFiles[0])
        }
      }
      // Reset input so the same file can be selected again
      e.target.value = ''
    }
  }

  const handleFolderChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files).filter(f => !shouldIgnoreFile(f))
      if (newFiles.length > 0) {
        // Append to existing files
        setFiles(prev => [...prev, ...newFiles])

        setResult(null)
        setError(null)
      }
      // Reset input so the same folder can be selected again
      e.target.value = ''
    }
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleFolderButtonClick = () => {
    folderInputRef.current?.click()
  }

  const handleProcess = async () => {
    if (files.length === 0) {
      setError('Please select a file or folder first')
      return
    }

    // Cleanup previous processed files from server if they exist
    if (result?.job_id) {
      try {
        await fetch(`${backendUrl}/api/cleanup/${result.job_id}`, {
          method: 'DELETE'
        })
      } catch (err) {
        console.error('Failed to cleanup previous files:', err)
      }
    }

    setProcessing(true)
    setError(null)
    setResult(null)
    setProcessingProgress({ current: 0, total: files.length })

    try {
      if (files.length === 1) {
        // Single file upload
        const formData = new FormData()
        formData.append('file', files[0])

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
      } else {
        // Multiple files upload
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

        const formData = new FormData()
        files.forEach((f) => {
          const relativePath = f.webkitRelativePath || f._relativePath || f.name
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
    } catch (err) {
      setError('Failed to download file: ' + err.message)
    }
  }

  const clearSelection = () => {
    setFiles([])
    setResult(null)
    setError(null)
    setFileInfo(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    if (folderInputRef.current) {
      folderInputRef.current.value = ''
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

    // Cleanup processed files from server if they exist
    if (result?.job_id) {
      try {
        await fetch(`${backendUrl}/api/cleanup/${result.job_id}`, {
          method: 'DELETE'
        })
      } catch (err) {
        console.error('Failed to cleanup files:', err)
      }
    }

    clearSelection()
  }

  return (
    <div className="file-upload">

      <button
        className="back-to-config-button"
        onClick={onBackToConfig}
        type="button"
      >
        <BsArrowLeft /> LLM Configuration
      </button>

      {/* Prompt customization */}
      <PromptSettings
        backendUrl={backendUrl}
        isOpen={promptSettingsOpen}
        onToggle={() => setPromptSettingsOpen(!promptSettingsOpen)}
      />

      {/* Video processing settings - show when video files are selected or DICOM video detected */}
      {(hasVideoFiles || checkingFileInfo) && (
        <div className="video-settings">
          <div className="video-settings-header">
            <BsCameraVideo className="video-icon" />
            <span className="video-settings-title">Video Processing Settings</span>
            {fileInfo?.is_video && fileInfo?.frame_count && (
              <span className="video-frame-count">
                ({fileInfo.frame_count} frames)
              </span>
            )}
          </div>
          {checkingFileInfo ? (
            <div className="video-settings-content">
              <span className="checking-file-info">Checking file format...</span>
            </div>
          ) : (
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
          )}
        </div>
      )}

      {/* Mapping files settings - show when folder is selected */}
      {files.length > 0 && (
        <div className="video-settings">
          <div className="video-settings-header">
            <BsFolder className="video-icon" />
            <span className="video-settings-title">Mapping Files Settings</span>
          </div>
          <div className="video-settings-content">
            <label className="video-checkbox-label">
              <input
                type="checkbox"
                className="video-checkbox"
                checked={saveMappingFiles}
                onChange={(e) => handleMappingFilesSettingChange(e.target.checked)}
                disabled={processing}
              />
              <div className="video-checkbox-info">
                <span className="video-checkbox-text">Save mapping CSV files</span>
                <span className="video-checkbox-description">
                  Create filename_anonymization.csv and folder_anonymization.csv files that map original names to anonymized names.
                </span>
              </div>
            </label>
          </div>
        </div>
      )}

      <div
        className={`drop-zone ${dragActive ? 'active' : ''}`}
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
          accept=".png,.jpg,.jpeg,.csv,.txt,.dcm,.dicom,.pdf,.hea,.docx,.xlsx,.xls,.mp4,.avi,.mov,.mkv,.wav,.mp3"
          multiple
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

        <div className="drop-zone-content">
          <div className="upload-icon"><BsUpload /></div>
          <p className="drop-zone-text">
            Drag and drop files or folders
          </p>
          <p className="drop-zone-subtext">or choose from your computer</p>
          <p className="supported-formats">
            Supports CSV, Excel, Word, Text, PDF, Images, DICOM, Video, Audio, and HEA files.
          </p>
          <div className="browse-buttons">
            <button
              type="button"
              className="browse-button browse-folder"
              onClick={handleFolderButtonClick}
              disabled={processing}
            >
              <BsFolder /> {files.length > 0 ? 'Add Folder' : 'Select Folder'}
            </button>
            <button
              type="button"
              className="browse-button"
              onClick={handleButtonClick}
              disabled={processing}
            >
              <BsFileEarmark /> {files.length > 0 ? 'Add Files' : 'Select Files'}
            </button>
          </div>
        </div>

        {files.length > 0 && (
          <div className="file-list-section">
            <div className="file-list-header">
              <span className="file-list-summary">
                {files.length} file{files.length !== 1 ? 's' : ''} ({formatFileSize(files.reduce((sum, f) => sum + f.size, 0))})
              </span>
              <button
                className="file-list-toggle"
                onClick={(e) => {
                  e.stopPropagation()
                  setFileListExpanded(prev => !prev)
                }}
                type="button"
              >
                {fileListExpanded ? <BsChevronUp /> : <BsChevronDown />}
              </button>
            </div>
            {fileListExpanded && (
              <div className="file-list-items">
                {files.map((f, idx) => (
                  <div key={idx} className="file-list-item">
                    <span className="file-list-icon"><BsFileEarmark /></span>
                    <span className="file-list-name">{f.webkitRelativePath || f._relativePath || f.name}</span>
                    <span className="file-list-size">{formatFileSize(f.size)}</span>
                    <button
                      className="file-list-remove"
                      onMouseDown={(e) => {
                        e.stopPropagation()
                        e.preventDefault()
                        const newFiles = files.filter((_, i) => i !== idx)
                        if (newFiles.length === 0) {
                          clearSelection()
                        } else {
                          setFiles(newFiles)
                        }
                      }}
                      disabled={processing}
                      type="button"
                      title="Remove file"
                      tabIndex={-1}
                    >
                      <BsX />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="file-list-actions">
              <button
                className="remove-files-button"
                onClick={(e) => {
                  e.stopPropagation()
                  clearSelection()
                }}
                disabled={processing}
                type="button"
              >
                Remove Files
              </button>
              {!result && (
                <button
                  className="upload-button"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleProcess()
                  }}
                  disabled={processing}
                  type="button"
                >
                  {processing
                    ? `Processing${processingProgress.total > 1 && processingProgress.current > 0 ? ` (${processingProgress.current}/${processingProgress.total})` : ''}...`
                    : 'Process Files'}
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="result-section">
          <div className={`success-message ${result.warnings?.length ? 'has-warnings' : ''}`}>
            {result.files_processed
              ? `${result.files_processed} files processed${result.file_results?.some(r => r.status === 'warning') ? ' (some with warnings)' : ' successfully'}!`
              : result.warnings?.length
                ? 'File processed with warnings'
                : 'File processed successfully!'}
            {result.files_failed > 0 && (
              <span className="warning-text"> ({result.files_failed} failed)</span>
            )}
          </div>

          {/* Show warnings for single file processing */}
          {result.warnings && result.warnings.length > 0 && (
            <div className="verification-warnings">
              <div className="verification-warning-header">
                <BsExclamationTriangle className="warning-icon" />
                <span>Processing Warnings</span>
              </div>
              <ul className="verification-warning-list">
                {result.warnings.map((warning, idx) => (
                  <li key={idx}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

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
                  <div key={idx} className={`batch-file-item ${fileResult.status === 'error' ? 'error' : ''} ${fileResult.status === 'warning' ? 'warning' : ''}`}>
                    <span className="batch-file-original">{fileResult.original_path}</span>
                    <span className="batch-file-arrow">→</span>
                    <span className="batch-file-anonymized">
                      {fileResult.status === 'error' ? fileResult.error : fileResult.anonymized_filename}
                    </span>
                    {fileResult.warnings && fileResult.warnings.length > 0 && (
                      <div className="batch-file-warnings">
                        <BsExclamationTriangle className="warning-icon-small" />
                        <span>{fileResult.warnings[0]}</span>
                      </div>
                    )}
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
          {funMessage && (
            <p className="processing-fun-message">{funMessage}</p>
          )}
        </div>
      )}
    </div>
  )
}

export default FileUpload
