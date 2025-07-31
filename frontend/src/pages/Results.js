import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './Results.css';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const analysisResult = location.state?.analysisResult;

  if (!analysisResult) {
    navigate('/');
    return null;
  }

  const handleNewAnalysis = () => {
    navigate('/');
  };

  const renderScoreCard = (title, score, maxScore, description) => (
    <div className="score-card">
      <div className="score-header">
        <h3>{title}</h3>
        <div className="score-display">
          <span className="score-value">{score}</span>
          <span className="score-max">/{maxScore}</span>
        </div>
      </div>
      <div className="score-bar">
        <div 
          className="score-fill" 
          style={{ width: `${(score / maxScore) * 100}%` }}
        ></div>
      </div>
      <p className="score-description">{description}</p>
    </div>
  );

  const getScoreColor = (score, maxScore) => {
    const percentage = (score / maxScore) * 100;
    if (percentage >= 80) return '#77ff99';
    if (percentage >= 60) return '#ffeb3b';
    if (percentage >= 40) return '#ff9800';
    return '#f44336';
  };

  return (
    <div className="results-container">
      <div className="results-header">
        <img src="/markaba-logo.png" alt="Markaba" className="logo" />
        <h1>Credit Analysis Results</h1>
        <button onClick={handleNewAnalysis} className="new-analysis-btn">
          New Analysis
        </button>
      </div>

      <div className="results-content">
        <div className="overall-score">
          <h2>Overall Credit Score</h2>
          <div className="main-score">
            <div 
              className="score-circle"
              style={{ 
                background: `conic-gradient(${getScoreColor(analysisResult.overall_score || 0, 850)} ${((analysisResult.overall_score || 0) / 850) * 360}deg, rgba(255,255,255,0.1) 0deg)`
              }}
            >
              <div className="score-inner">
                <span className="main-score-value">{analysisResult.overall_score || 'N/A'}</span>
                <span className="main-score-label">Credit Score</span>
              </div>
            </div>
          </div>
        </div>

        <div className="detailed-scores">
          {analysisResult.payment_history && renderScoreCard(
            "Payment History",
            analysisResult.payment_history,
            100,
            "Your track record of making payments on time"
          )}
          
          {analysisResult.credit_utilization && renderScoreCard(
            "Credit Utilization",
            Math.round(analysisResult.credit_utilization),
            100,
            "Percentage of available credit you're using"
          )}
          
          {analysisResult.credit_age && renderScoreCard(
            "Credit Age",
            analysisResult.credit_age,
            100,
            "Average age of your credit accounts"
          )}
          
          {analysisResult.credit_mix && renderScoreCard(
            "Credit Mix",
            analysisResult.credit_mix,
            100,
            "Variety of credit account types you have"
          )}
        </div>

        {analysisResult.recommendations && (
          <div className="recommendations">
            <h3>💡 Recommendations</h3>
            <div className="recommendations-list">
              {analysisResult.recommendations.map((recommendation, index) => (
                <div key={index} className="recommendation-item">
                  <div className="recommendation-icon">✓</div>
                  <p>{recommendation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {analysisResult.summary && (
          <div className="summary-section">
            <h3>📊 Analysis Summary</h3>
            <div className="summary-content">
              <p>{analysisResult.summary}</p>
            </div>
          </div>
        )}

        {analysisResult.raw_data && (
          <div className="raw-data-section">
            <h3>📋 Detailed Information</h3>
            <div className="raw-data-content">
              <pre>{JSON.stringify(analysisResult.raw_data, null, 2)}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Results;
