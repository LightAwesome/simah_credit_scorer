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

  // Handle both old and new result formats
  const getCalculationData = () => {
    // New format from the calculation endpoint
    if (analysisResult.calculation_result && analysisResult.calculation_result.success) {
      return analysisResult.calculation_result;
    }
    // Legacy format
    return analysisResult;
  };

  const calculationData = getCalculationData();
  const finalResult = calculationData.final_result || {};
  const sections = calculationData.sections || [];

  const renderScoreCard = (title, score, maxScore, description, details = null) => (
    <div className="score-card">
      <div className="score-header">
        <h3>{title}</h3>
        <div className="score-display">
          <span className="score-value">{Math.round(score)}</span>
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
      {details && (
        <div className="score-details">
          {details}
        </div>
      )}
    </div>
  );

  const renderFormulaDetails = (calculation) => (
    <div className="formula-details">
      <div className="formula-header">
        <span className="formula-name">{calculation.name}</span>
        <span className="formula-score">{Math.round(calculation.score)}/{calculation.max_points}</span>
      </div>
      <div className="formula-info">
        <p><strong>Formula:</strong> {calculation.formula}</p>
        <p><strong>Weight:</strong> {calculation.weight}%</p>
        {calculation.variables_used && Object.keys(calculation.variables_used).length > 0 && (
          <div className="variables-used">
            <strong>Variables:</strong>
            <ul>
              {Object.entries(calculation.variables_used).map(([key, value]) => (
                <li key={key}>{key}: {value}</li>
              ))}
            </ul>
          </div>
        )}
        {calculation.missing_variables && calculation.missing_variables.length > 0 && (
          <div className="missing-variables">
            <strong>Missing Variables:</strong> {calculation.missing_variables.join(', ')}
          </div>
        )}
      </div>
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
                background: `conic-gradient(${getScoreColor(finalResult.final_credit_score || 0, 900)} ${((finalResult.final_credit_score || 0) / 900) * 360}deg, rgba(255,255,255,0.1) 0deg)`
              }}
            >
              <div className="score-inner">
                <span className="main-score-value">{Math.round(finalResult.final_credit_score || 0)}</span>
                <span className="main-score-label">Credit Score</span>
                <span className="score-percentage">{Math.round(finalResult.score_percentage || 0)}%</span>
              </div>
            </div>
          </div>
          {finalResult.total_weighted_score && (
            <div className="score-breakdown">
              <p>Total Weighted Score: {Math.round(finalResult.total_weighted_score)}</p>
              <p>Total Possible Score: {Math.round(finalResult.total_possible_score || 0)}</p>
            </div>
          )}
        </div>

        {/* Section-based Results */}
        {sections.length > 0 && (
          <div className="detailed-scores">
            <h3>📊 Section Breakdown</h3>
            {sections.map((section, index) => (
              <div key={index} className="section-results">
                {renderScoreCard(
                  section.name,
                  section.weighted_score || 0,
                  section.weight || 100,
                  `Weight: ${section.weight}% of total score`,
                  <div className="section-calculations">
                    {section.calculations && section.calculations.map((calc, calcIndex) => (
                      <div key={calcIndex} className="calculation-result">
                        {renderFormulaDetails(calc)}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Legacy Results (fallback) */}
        {sections.length === 0 && (
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
        )}

        {/* Data Analysis Summary */}
        {calculationData.data_analysis && (
          <div className="data-analysis">
            <h3>📋 Data Analysis</h3>
            <div className="analysis-stats">
              <p><strong>Variables Available:</strong> {calculationData.data_analysis.variables_available}</p>
              {calculationData.data_analysis.variables_list && (
                <details className="variables-details">
                  <summary>View Available Variables ({calculationData.data_analysis.variables_list.length})</summary>
                  <div className="variables-list">
                    {calculationData.data_analysis.variables_list.map((variable, index) => (
                      <span key={index} className="variable-tag">{variable}</span>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        )}

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

        {/* Show calculation errors if any */}
        {analysisResult.calculation_error && (
          <div className="error-section">
            <h3>⚠️ Calculation Issues</h3>
            <div className="error-content">
              <p>There was an issue with the calculation: {analysisResult.calculation_error}</p>
              <p>The data was extracted successfully, but the credit score calculation encountered problems.</p>
            </div>
          </div>
        )}

        {/* Raw extracted data for debugging */}
        {analysisResult.extracted_data && (
          <div className="raw-data-section">
            <h3>📋 Extracted Data</h3>
            <details className="raw-data-content">
              <summary>View Raw Extracted Data</summary>
              <pre>{JSON.stringify(analysisResult.extracted_data, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
};

export default Results;
