import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './Upload.css';

const Upload = () => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
    } else {
      alert('Please select a PDF file');
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
      } else {
        alert('Please select a PDF file');
      }
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      alert('Please select a PDF file first');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        // Show success message instead of navigating to results
        alert(`File uploaded successfully! Data extracted: ${JSON.stringify(result.extracted_data, null, 2)}`);
        // Reset file selection
        setFile(null);
      } else {
        alert('Error analyzing file. Please try again.');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-header">
        <div className="header-top">
          <img src="/markaba-logo.png" alt="Markaba" className="logo" />
          <Link to="/configure" className="configure-link">
            ⚙️ Configure Calculations
          </Link>
        </div>
        <h1>Simah Credit Scorer</h1>
        <p>Upload your credit report PDF to get instant analysis</p>
      </div>

      <div className="upload-card">
        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="upload-icon">📄</div>
          <h3>Drop your PDF here or click to browse</h3>
          <p>Only PDF files are accepted</p>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="file-input"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="file-label">
            Choose File
          </label>
        </div>

        {file && (
          <div className="file-info">
            <div className="file-details">
              <span className="file-name">📄 {file.name}</span>
              <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <button 
              onClick={() => setFile(null)} 
              className="remove-file"
            >
              ✕
            </button>
          </div>
        )}

        <button 
          onClick={handleAnalyze}
          disabled={!file || isUploading}
          className="analyze-button"
        >
          {isUploading ? 'Analyzing...' : 'Analyze Credit Report'}
        </button>
      </div>
    </div>
  );
};

export default Upload;
