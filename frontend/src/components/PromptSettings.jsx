import { useState, useEffect } from 'react'
import './PromptSettings.css'
import { BsChevronDown, BsChevronUp, BsArrowCounterclockwise } from 'react-icons/bs'

function PromptSettings({ backendUrl, isOpen, onToggle }) {
  const [prompts, setPrompts] = useState(null)
  const [descriptions, setDescriptions] = useState({})
  const [templateVariables, setTemplateVariables] = useState({})
  const [defaults, setDefaults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [expandedPrompt, setExpandedPrompt] = useState(null)
  const [expandedGroups, setExpandedGroups] = useState(new Set())

  // Prompt field labels for display
  const promptLabels = {
    column_detection_prompt: 'Column Detection',
    csv_anonymization_prompt: 'PII Anonymization',
    text_anonymization_prompt: 'PII Anonymization',
    csv_verification_prompt: 'Verification',
    text_verification_prompt: 'Verification',
    image_anonymization_prompt: 'PII Anonymization',
    image_verification_prompt: 'Verification',
    pdf_anonymization_prompt: 'PII Anonymization',
    pdf_verification_prompt: 'Verification',
    dicom_metadata_anonymization_prompt: 'DICOM Metadata',
    additional_instructions: 'Additional Instructions'
  }

  // Group prompts by file type
  const promptGroups = {
    csv: ['column_detection_prompt', 'csv_anonymization_prompt', 'csv_verification_prompt'],
    text: ['text_anonymization_prompt', 'text_verification_prompt'],
    image: ['image_anonymization_prompt', 'image_verification_prompt', 'dicom_metadata_anonymization_prompt'],
    pdf: ['pdf_anonymization_prompt', 'pdf_verification_prompt']
  }

  const groupLabels = {
    csv: 'Tabular Files',
    text: 'Text Files',
    image: 'Images',
    pdf: 'PDF Files'
  }

  useEffect(() => {
    if (isOpen) {
      fetchPrompts()
      fetchDefaults()
    }
  }, [isOpen, backendUrl])

  const fetchPrompts = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${backendUrl}/api/prompts`)
      if (!response.ok) throw new Error('Failed to fetch prompts')
      const data = await response.json()
      setPrompts(data.prompts)
      setDescriptions(data.descriptions)
      if (data.template_variables) setTemplateVariables(data.template_variables)
      setError(null)
    } catch (err) {
      setError('Failed to load prompt settings: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchDefaults = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/prompts/defaults`)
      if (!response.ok) throw new Error('Failed to fetch defaults')
      const data = await response.json()
      setDefaults(data.prompts)
    } catch (err) {
      console.error('Failed to load defaults:', err)
    }
  }

  const handlePromptChange = (key, value) => {
    setPrompts(prev => ({
      ...prev,
      [key]: value
    }))
    setSuccess(null)
  }

  const handleSave = async () => {
    try {
      setSaving(true)
      setError(null)
      setSuccess(null)

      const response = await fetch(`${backendUrl}/api/prompts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(prompts),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to save prompts')
      }

      const data = await response.json()
      setPrompts(data.prompts)
      setSuccess('Prompts saved successfully')
    } catch (err) {
      setError('Failed to save: ' + err.message)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    if (!window.confirm('Reset all prompts to their default values?')) {
      return
    }

    try {
      setSaving(true)
      setError(null)
      setSuccess(null)

      const response = await fetch(`${backendUrl}/api/prompts/reset`, {
        method: 'POST',
      })

      if (!response.ok) throw new Error('Failed to reset prompts')

      const data = await response.json()
      setPrompts(data.prompts)
      setSuccess('Prompts reset to defaults')
    } catch (err) {
      setError('Failed to reset: ' + err.message)
    } finally {
      setSaving(false)
    }
  }

  const handleResetSingle = (key) => {
    if (defaults && defaults[key]) {
      handlePromptChange(key, defaults[key])
    }
  }

  const isModified = (key) => {
    if (!defaults || !prompts) return false
    return prompts[key] !== defaults[key]
  }

  const hasAnyModified = () => {
    if (!defaults || !prompts) return false
    return Object.keys(defaults).some(key => prompts[key] !== defaults[key])
  }

  const hasGroupModified = (groupKey) => {
    if (!defaults || !prompts) return false
    const keys = promptGroups[groupKey] || []
    return keys.some(key => prompts[key] !== defaults[key])
  }

  const togglePrompt = (key) => {
    setExpandedPrompt(expandedPrompt === key ? null : key)
  }

  const toggleGroup = (groupKey) => {
    setExpandedGroups(prev => {
      const newSet = new Set(prev)
      if (newSet.has(groupKey)) {
        newSet.delete(groupKey)
      } else {
        newSet.add(groupKey)
      }
      return newSet
    })
  }

  if (!isOpen) {
    return (
      <div className="prompt-settings-collapsed" onClick={onToggle}>
        <span className="prompt-settings-toggle">
          <BsChevronDown /> Customize Prompts
          {hasAnyModified() && <span className="prompt-modified-badge">Modified</span>}
        </span>
      </div>
    )
  }

  return (
    <div className="prompt-settings">
      <div className="prompt-settings-header" onClick={onToggle}>
        <h3>
          <BsChevronUp /> Customize Prompts
        </h3>
        <p className="prompt-settings-subtitle">
          Customize how the AI detects and redacts personal information. Each prompt controls a specific step of the anonymization process.
        </p>
      </div>

      {loading ? (
        <div className="prompt-settings-loading">Loading prompt settings...</div>
      ) : error ? (
        <div className="prompt-settings-error">{error}</div>
      ) : prompts ? (
        <div className="prompt-settings-content">
          <div className={`prompt-field ${isModified('additional_instructions') ? 'prompt-field-modified' : ''}`}>
            <div
              className="prompt-field-header prompt-field-collapsible"
              onClick={() => togglePrompt('additional_instructions')}
            >
              <label>
                {expandedPrompt === 'additional_instructions' ? <BsChevronUp /> : <BsChevronDown />}
                {promptLabels.additional_instructions}
                {isModified('additional_instructions') && <span className="prompt-modified-dot" title="Modified from default" />}
              </label>
              <button
                className="prompt-reset-single"
                onClick={(e) => {
                  e.stopPropagation()
                  handleResetSingle('additional_instructions')
                }}
                title="Reset to default"
              >
                <BsArrowCounterclockwise />
              </button>
            </div>
            {expandedPrompt === 'additional_instructions' && (
              <>
                <p className="prompt-description">{descriptions.additional_instructions}</p>
                <textarea
                  value={prompts.additional_instructions || ''}
                  onChange={(e) => handlePromptChange('additional_instructions', e.target.value)}
                  placeholder="Add custom instructions that will be appended to all prompts..."
                  rows={3}
                />
              </>
            )}
          </div>

          {/* Prompts grouped by file type */}
          {Object.entries(promptGroups).map(([groupKey, promptKeys]) => {
            const isGroupExpanded = expandedGroups.has(groupKey)
            return (
              <div key={groupKey} className="prompt-group">
                <h4
                  className="prompt-group-title prompt-group-title-collapsible"
                  onClick={() => toggleGroup(groupKey)}
                >
                  {isGroupExpanded ? <BsChevronUp /> : <BsChevronDown />}
                  {groupLabels[groupKey]}
                  {hasGroupModified(groupKey) && <span className="prompt-modified-dot" title="Contains modified prompts" />}
                </h4>
                {isGroupExpanded && promptKeys.map((key) => {
                  const value = prompts[key]
                  return (
                    <div key={key} className={`prompt-field ${isModified(key) ? 'prompt-field-modified' : ''}`}>
                      <div
                        className="prompt-field-header prompt-field-collapsible"
                        onClick={() => togglePrompt(key)}
                      >
                        <label>
                          {expandedPrompt === key ? <BsChevronUp /> : <BsChevronDown />}
                          {promptLabels[key] || key}
                          {isModified(key) && <span className="prompt-modified-dot" title="Modified from default" />}
                        </label>
                        <button
                          className="prompt-reset-single"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleResetSingle(key)
                          }}
                          title="Reset to default"
                        >
                          <BsArrowCounterclockwise />
                        </button>
                      </div>
                      {expandedPrompt === key && (
                        <>
                          <p className="prompt-description">{descriptions[key]}</p>
                          {templateVariables[key] && templateVariables[key].length > 0 && (
                            <div className="prompt-variables-info">
                              <span className="prompt-variables-label">Required placeholders (do not remove):</span>
                              <span className="prompt-variables-list">
                                {templateVariables[key].map((v) => (
                                  <code key={v} className={
                                    'prompt-variable-tag' +
                                    (value && !value.includes(v) ? ' prompt-variable-missing' : '')
                                  }>{v}</code>
                                ))}
                              </span>
                            </div>
                          )}
                          <textarea
                            value={value || ''}
                            onChange={(e) => handlePromptChange(key, e.target.value)}
                            rows={12}
                          />
                        </>
                      )}
                    </div>
                  )
                })}
              </div>
            )
          })}

          <div className="prompt-settings-actions">
            <button
              className="prompt-save-button"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
            <button
              className="prompt-reset-button"
              onClick={handleReset}
              disabled={saving}
            >
              Reset All to Defaults
            </button>
          </div>

          {success && <div className="prompt-success">{success}</div>}
        </div>
      ) : null}
    </div>
  )
}

export default PromptSettings
