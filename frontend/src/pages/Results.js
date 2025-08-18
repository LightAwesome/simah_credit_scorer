import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './Results.css';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const analysisResult = location.state?.analysisResult;

  // Add debugging
  console.log('Analysis Result:', analysisResult);
  console.log('Location state:', location.state);

  if (!analysisResult) {
    console.log('No analysis result found, redirecting to home');
    navigate('/');
    return null;
  }

  // Additional validation
  if (!analysisResult.success) {
    console.log('Analysis result indicates failure:', analysisResult);
    alert(`Analysis failed: ${analysisResult.error || analysisResult.calculation_error || 'Unknown error'}`);
    navigate('/');
    return null;
  }

  const handleNewAnalysis = () => {
    navigate('/');
  };

  // Handle both old and new result formats
  const getCalculationData = () => {
    // New simplified format with overall_score and sections directly
    if (analysisResult.overall_score !== undefined && analysisResult.sections) {
      return {
        overall_score: analysisResult.overall_score,
        sections: analysisResult.sections
      };
    }
    // Legacy format from the LLM calculation endpoint
    if (analysisResult.calculation_result && analysisResult.calculation_result.success) {
      return analysisResult.calculation_result;
    }
    // Legacy format
    return analysisResult;
  };

  // Parse LLM results if they exist
  const parseLLMResults = (calculationData) => {
    // New simplified format with overall_score and sections directly
    if (calculationData.overall_score !== undefined && calculationData.sections) {
      return {
        overall_score: calculationData.overall_score,
        sections: calculationData.sections
      };
    }
    
    // Legacy format - check if we have parsed results already
    if (calculationData.results && Array.isArray(calculationData.results)) {
      return calculationData.results;
    }
    
    // Legacy handling for string results
    if (calculationData.results && typeof calculationData.results === 'string') {
      try {
        // Try to parse the LLM output as JSON
        const parsed = JSON.parse(calculationData.results);
        // Handle both array and single object cases
        if (Array.isArray(parsed)) {
          return parsed;
        } else if (typeof parsed === 'object' && parsed !== null) {
          return [parsed]; // Wrap single object in array
        }
        return [];
      } catch (error) {
        console.error('Error parsing LLM results:', error);
        // If JSON parsing fails, try to extract JSON from the string
        const jsonMatch = calculationData.results.match(/\[.*\]/s);
        if (jsonMatch) {
          try {
            return JSON.parse(jsonMatch[0]);
          } catch (e) {
            console.error('Error parsing extracted JSON:', e);
            // Try to extract single object
            const singleObjectMatch = calculationData.results.match(/\{.*\}/s);
            if (singleObjectMatch) {
              try {
                const singleResult = JSON.parse(singleObjectMatch[0]);
                return [singleResult];
              } catch (e2) {
                console.error('Error parsing single object:', e2);
                return [];
              }
            }
            return [];
          }
        }
        return [];
      }
    }
    return [];
  };

  const calculationData = getCalculationData();
  const llmResults = parseLLMResults(calculationData);
  const finalResult = calculationData.final_result || {};

  // Add debugging for calculation data
  console.log('Calculation Data:', calculationData);
  console.log('LLM Results:', llmResults);

  // Get overall score from new structure or legacy
  const getOverallScore = () => {
    if (analysisResult.overall_score !== undefined) {
      return analysisResult.overall_score;
    }
    if (llmResults && llmResults.overall_score !== undefined) {
      return llmResults.overall_score;
    }
    return finalResult.final_credit_score || 0;
  };

  // Safe access to final metrics
  const getFinalMetrics = () => {
    return analysisResult.final_metrics || {};
  };

  const getSimahScore = () => {
    const metrics = getFinalMetrics();
    return metrics.simah_credit_score || 0;
  };

  const renderLLMResults = () => {
    // Handle new structured format
    if (llmResults && llmResults.sections && Array.isArray(llmResults.sections)) {
      return (
        <div className="llm-results-section">
          <h2>AI Calculation Results</h2>
          <div className="llm-sections-grid">
            {llmResults.sections.map((section, index) => (
              <div key={index} className="llm-section-card">
                <div className="section-header">
                  <h3 className="section-title">{section.name}</h3>
                  <div className="section-total">
                    <span className="total-label">Total:</span>
                    <span className="total-value">{section.total || 'N/A'}</span>
                  </div>
                </div>
                <div className="section-formulas">
                  {section.formulas && section.formulas.map((formula, formulaIndex) => (
                    <div key={formulaIndex} className="formula-result">
                      <div className="formula-name">{formula.name}</div>
                      <div className="formula-value">{formula.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      );
    }

    // Handle legacy format
    if (llmResults && Array.isArray(llmResults) && llmResults.length > 0) {
      return (
        <div className="llm-results-section">
          <h2>AI Calculation Results</h2>
          <div className="llm-results-grid">
            {llmResults.map((result, index) => (
              <div key={index} className="llm-result-card">
                {Object.entries(result).map(([formula, value]) => (
                  <div key={formula} className="formula-result">
                    <div className="formula-name">{formula}</div>
                    <div className="formula-value">{value}</div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      );
    }

    return null;
  };



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
          <h2>Simah Credit Score</h2>
          <div className="main-score">
            <div 
              className="score-circle"
              style={{ 
                background: `conic-gradient(${getScoreColor(getSimahScore(), 850)} ${(getSimahScore() - 300) / 550 * 360}deg, rgba(255,255,255,0.1) 0deg)`
              }}
            >
              <div className="score-inner">
                <span className="main-score-value">{Math.round(getSimahScore())}</span>
                <span className="main-score-label">Simah Credit Score</span>
                <span className="score-percentage">{Math.round((getSimahScore() - 300) / 550 * 100)}%</span>
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

        {/* Final Metrics Section */}
        {Object.keys(getFinalMetrics()).length > 0 && (
          <div className="final-metrics-section">
            <h2>Final Credit Assessment</h2>
            <div className="metrics-grid">
              <div className="metric-card weighted-score">
                <div className="metric-header">
                  <h3>Weighted Score</h3>
                  <div className="metric-value">
                    {Math.round(getOverallScore())}
                  </div>
                </div>
                <div className="metric-range">Range: 0-1020</div>
                <div className="metric-bar">
                  <div 
                    className="metric-fill weighted-fill" 
                    style={{ 
                      width: `${(getOverallScore() / 1020) * 100}%` 
                    }}
                  ></div>
                </div>
              </div>

              <div className="metric-card risk-grade">
                <div className="metric-header">
                  <h3>Risk Grade</h3>
                  <div className={`metric-value grade-${getFinalMetrics().risk_grade || 'F'}`}>
                    {getFinalMetrics().risk_grade || 'F'}
                  </div>
                </div>
                <div className="metric-range">Grades: A (Best) - F (Highest Risk)</div>
                <div className="grade-scale">
                  {['A', 'B', 'C', 'D', 'E', 'F'].map(grade => (
                    <span 
                      key={grade} 
                      className={`grade-indicator ${grade === getFinalMetrics().risk_grade ? 'active' : ''}`}
                    >
                      {grade}
                    </span>
                  ))}
                </div>
              </div>

              <div className="metric-card interest-rate">
                <div className="metric-header">
                  <h3>Interest Rate</h3>
                  <div className="metric-value">
                    {(getFinalMetrics().interest_rate || 0).toFixed(1)}%
                  </div>
                </div>
                <div className="metric-range">Typical Range: 6.5% - 11.5%</div>
                <div className="metric-bar">
                  <div 
                    className="metric-fill interest-fill" 
                    style={{ 
                      width: `${Math.min(((getFinalMetrics().interest_rate || 0) - 6.5) / 5 * 100, 100)}%` 
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* LLM Calculation Results */}
        {llmResults && renderLLMResults()}

      </div>
    </div>
  );
};

export default Results;
