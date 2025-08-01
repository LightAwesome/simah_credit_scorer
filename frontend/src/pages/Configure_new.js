import React, { useState, useEffect } from 'react';
import './Configure.css';
import calculationsData from '../data/calculations.json';

const Configure = () => {
  const [calculations, setCalculations] = useState(calculationsData);
  const [editingFormula, setEditingFormula] = useState(null);
  const [showAddSection, setShowAddSection] = useState(false);
  const [showAddFormula, setShowAddFormula] = useState(null);
  const [selectedVariables, setSelectedVariables] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const handleEditFormula = (sectionIndex, calculationIndex) => {
    setEditingFormula({ sectionIndex, calculationIndex });
  };

  const handleDeleteFormula = (sectionIndex, calculationIndex) => {
    if (window.confirm('Are you sure you want to delete this formula?')) {
      const updatedCalculations = { ...calculations };
      updatedCalculations.sections[sectionIndex].calculations.splice(calculationIndex, 1);
      setCalculations(updatedCalculations);
    }
  };

  const handleAddFormula = (sectionIndex) => {
    setShowAddFormula(sectionIndex);
  };

  const handleDeleteSection = (sectionIndex) => {
    if (window.confirm('Are you sure you want to delete this entire section?')) {
      const updatedCalculations = { ...calculations };
      updatedCalculations.sections.splice(sectionIndex, 1);
      setCalculations(updatedCalculations);
    }
  };

  const handleSaveFormula = (sectionIndex, calculationIndex, updatedCalculation) => {
    const updatedCalculations = { ...calculations };
    if (calculationIndex === -1) {
      // Adding new formula
      updatedCalculations.sections[sectionIndex].calculations.push(updatedCalculation);
    } else {
      // Updating existing formula
      updatedCalculations.sections[sectionIndex].calculations[calculationIndex] = updatedCalculation;
    }
    setCalculations(updatedCalculations);
    setEditingFormula(null);
    setShowAddFormula(null);
  };

  const handleAddSection = (newSection) => {
    const updatedCalculations = { ...calculations };
    updatedCalculations.sections.push(newSection);
    setCalculations(updatedCalculations);
    setShowAddSection(false);
  };

  const saveCalculations = async () => {
    try {
      const response = await fetch('/api/calculations', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(calculations),
      });
      
      if (response.ok) {
        alert('Calculations saved successfully!');
      } else {
        alert('Error saving calculations');
      }
    } catch (error) {
      console.error('Error saving calculations:', error);
      alert('Error saving calculations');
    }
  };

  // Variable manipulation functions
  const toggleVariableSelection = (variableName) => {
    setSelectedVariables(prev => 
      prev.includes(variableName) 
        ? prev.filter(v => v !== variableName)
        : [...prev, variableName]
    );
  };

  const clearSelectedVariables = () => {
    setSelectedVariables([]);
  };

  // Filter variables based on search and category
  const getFilteredVariables = () => {
    let filteredVars = {};
    
    Object.entries(calculations.variables).forEach(([category, variables]) => {
      if (selectedCategory === 'all' || selectedCategory === category) {
        const filtered = variables.filter(variable =>
          variable.toLowerCase().includes(searchTerm.toLowerCase())
        );
        if (filtered.length > 0) {
          filteredVars[category] = filtered;
        }
      }
    });
    
    return filteredVars;
  };

  // Variable Selector Component
  const VariableSelector = () => {
    const filteredVariables = getFilteredVariables();
    const categoryColors = {
      payment: '#ff6b6b',
      balance: '#4ecdc4', 
      account: '#45b7d1',
      date: '#96ceb4',
      inquiry: '#feca57',
      score: '#ff9ff3',
      personal: '#54a0ff',
      employment: '#5f27cd',
      contact: '#00d2d3'
    };

    return (
      <div className="variable-selector">
        <div className="selector-header">
          <h3>📚 Variables Library</h3>
          <div className="search-controls">
            <input
              type="text"
              placeholder="🔍 Search variables..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="variable-search"
            />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="category-filter"
            >
              <option value="all">All Categories</option>
              {Object.keys(calculations.variables).map(category => (
                <option key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="variables-container">
          {Object.entries(filteredVariables).map(([category, variables]) => (
            <div key={category} className="category-section">
              <div className="category-header">
                <h4 className="category-title">{category.toUpperCase()}</h4>
                <span className="category-count">({variables.length})</span>
              </div>
              
              <div className="variables-list">
                {variables.map((variable, index) => (
                  <div
                    key={`${category}-${index}`}
                    className={`variable-card ${selectedVariables.includes(variable) ? 'selected' : ''}`}
                    draggable
                    onClick={() => toggleVariableSelection(variable)}
                    onDragStart={(e) => {
                      e.dataTransfer.setData('text/plain', `{${variable.toLowerCase().replace(/\s+/g, '_')}}`);
                      e.dataTransfer.setData('variable-name', variable);
                    }}
                  >
                    <div 
                      className="category-indicator"
                      style={{ backgroundColor: categoryColors[category] }}
                    ></div>
                    <span className="variable-name">{variable}</span>
                    {selectedVariables.includes(variable) && (
                      <span className="selected-indicator">✓</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Calculations Table Component
  const CalculationsTable = () => (
    <div className="calculations-table">
      <div className="table-header">
        <div className="header-cell">Section</div>
        <div className="header-cell">Weight %</div>
        <div className="header-cell">Formula</div>
        <div className="header-cell">Weight</div>
        <div className="header-cell">Max</div>
        <div className="header-cell">Actions</div>
      </div>

      {calculations.sections.map((section, sectionIndex) => (
        <div key={sectionIndex} className="section-group">
          {section.calculations.map((calculation, calculationIndex) => (
            <div key={calculationIndex} className="table-row">
              <div className="table-cell section-name">
                {calculationIndex === 0 ? (
                  <div className="section-info">
                    <span className="section-title">{section.name}</span>
                    <button 
                      className="delete-section-btn"
                      onClick={() => handleDeleteSection(sectionIndex)}
                      title="Delete Section"
                    >
                      🗑️
                    </button>
                  </div>
                ) : (
                  <div className="section-continuation">***</div>
                )}
              </div>
              
              <div className="table-cell section-percentage">
                {calculationIndex === 0 ? `${section.weight}%` : '***'}
              </div>
              
              <div className="table-cell formula">
                <div className="formula-content">
                  <span className="formula-name">{calculation.name}</span>
                  <div 
                    className="formula-drop-zone"
                    onClick={() => handleEditFormula(sectionIndex, calculationIndex)}
                    onDrop={(e) => {
                      e.preventDefault();
                      handleEditFormula(sectionIndex, calculationIndex);
                    }}
                    onDragOver={(e) => e.preventDefault()}
                  >
                    <span className="formula-text">{calculation.formula}</span>
                    <div className="drop-hint">Drop variables here or click to edit</div>
                  </div>
                </div>
              </div>
              
              <div className="table-cell variable-weight">
                {calculation.weight}%
              </div>
              
              <div className="table-cell max-points">
                {calculation.max_points}
              </div>
              
              <div className="table-cell actions">
                <button 
                  className="edit-btn"
                  onClick={() => handleEditFormula(sectionIndex, calculationIndex)}
                  title="Edit Formula"
                >
                  ✏️
                </button>
                <button 
                  className="delete-btn"
                  onClick={() => handleDeleteFormula(sectionIndex, calculationIndex)}
                  title="Delete Formula"
                >
                  🗑️
                </button>
              </div>
            </div>
          ))}
          
          <div className="add-formula-row">
            <div className="table-cell"></div>
            <div className="table-cell"></div>
            <div className="table-cell formula">
              <button 
                className="add-formula-btn"
                onClick={() => handleAddFormula(sectionIndex)}
              >
                + Add Formula to {section.name}
              </button>
            </div>
            <div className="table-cell"></div>
            <div className="table-cell"></div>
            <div className="table-cell"></div>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="configure-container">
      <div className="configure-header">
        <img src="/markaba-logo.png" alt="Markaba" className="logo" />
        <h1>Configure Credit Scoring</h1>
        <div className="header-buttons">
          <button className="add-section-button" onClick={() => setShowAddSection(true)}>
            + Add Section
          </button>
          <button className="save-button" onClick={saveCalculations}>
            Save Changes
          </button>
        </div>
      </div>

      <div className="main-layout">
        {/* Variables Library - Left Side */}
        <div className="variables-sidebar">
          <VariableSelector />
        </div>

        {/* Calculations Table - Right Side */}
        <div className="calculations-panel">
          <CalculationsTable />
        </div>
      </div>

      {/* Edit Formula Modal */}
      {editingFormula && (
        <FormulaEditor
          section={calculations.sections[editingFormula.sectionIndex]}
          calculation={calculations.sections[editingFormula.sectionIndex].calculations[editingFormula.calculationIndex]}
          variables={calculations.variables}
          onSave={(updatedCalculation) => handleSaveFormula(editingFormula.sectionIndex, editingFormula.calculationIndex, updatedCalculation)}
          onCancel={() => setEditingFormula(null)}
        />
      )}

      {/* Add Formula Modal */}
      {showAddFormula !== null && (
        <FormulaEditor
          section={calculations.sections[showAddFormula]}
          calculation={null}
          variables={calculations.variables}
          onSave={(newCalculation) => handleSaveFormula(showAddFormula, -1, newCalculation)}
          onCancel={() => setShowAddFormula(null)}
        />
      )}

      {/* Add Section Modal */}
      {showAddSection && (
        <SectionEditor
          onSave={handleAddSection}
          onCancel={() => setShowAddSection(false)}
        />
      )}
    </div>
  );
};

// Formula Editor Component
const FormulaEditor = ({ section, calculation, variables, onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    name: calculation?.name || '',
    formula: calculation?.formula || '',
    weight: calculation?.weight || 0,
    max_points: calculation?.max_points || 0,
    variables: calculation?.variables || []
  });
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedVariable = e.dataTransfer.getData('text/plain');
    const variableName = e.dataTransfer.getData('variable-name');
    
    // Insert at cursor position or append
    const textarea = e.target;
    const cursorPosition = textarea.selectionStart;
    const currentFormula = formData.formula;
    const newFormula = 
      currentFormula.slice(0, cursorPosition) + 
      droppedVariable + 
      currentFormula.slice(cursorPosition);
    
    setFormData(prev => ({
      ...prev,
      formula: newFormula
    }));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const insertOperator = (operator) => {
    setFormData(prev => ({
      ...prev,
      formula: prev.formula + ` ${operator} `
    }));
  };

  const insertFunction = (func) => {
    setFormData(prev => ({
      ...prev,
      formula: prev.formula + `${func}()`
    }));
  };

  const handleSave = () => {
    if (!formData.name || !formData.formula) {
      alert('Please fill in all required fields');
      return;
    }
    onSave(formData);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content formula-editor">
        <h3>{calculation ? 'Edit Formula' : 'Add New Formula'}</h3>
        
        <div className="form-group">
          <label>Formula Name:</label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
            placeholder="e.g., Payment History Score"
          />
        </div>

        <div className="form-group">
          <label>Formula Builder:</label>
          <div className="formula-builder">
            <div className="formula-operators">
              <button type="button" onClick={() => insertOperator('+')}>+</button>
              <button type="button" onClick={() => insertOperator('-')}>-</button>
              <button type="button" onClick={() => insertOperator('*')}>×</button>
              <button type="button" onClick={() => insertOperator('/')}>/</button>
              <button type="button" onClick={() => insertOperator('(')}>(</button>
              <button type="button" onClick={() => insertOperator(')')}>)</button>
              <button type="button" onClick={() => insertFunction('IF')}>IF</button>
              <button type="button" onClick={() => insertFunction('MIN')}>MIN</button>
              <button type="button" onClick={() => insertFunction('MAX')}>MAX</button>
            </div>
            <textarea
              className={`formula-input ${dragOver ? 'drag-over' : ''}`}
              value={formData.formula}
              onChange={(e) => setFormData(prev => ({ ...prev, formula: e.target.value }))}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              placeholder="Drag variables here or type formula... e.g., ({credit_score} / 900) * 200"
              rows="4"
            />
            <div className="formula-help">
              💡 Tip: Drag variables from the library or use the operator buttons
            </div>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Weight (%):</label>
            <input
              type="number"
              value={formData.weight}
              onChange={(e) => setFormData(prev => ({ ...prev, weight: parseInt(e.target.value) || 0 }))}
              min="0"
              max="100"
            />
          </div>

          <div className="form-group">
            <label>Max Points:</label>
            <input
              type="number"
              value={formData.max_points}
              onChange={(e) => setFormData(prev => ({ ...prev, max_points: parseInt(e.target.value) || 0 }))}
              min="0"
            />
          </div>
        </div>

        <div className="modal-buttons">
          <button onClick={handleSave} className="save-btn">Save Formula</button>
          <button onClick={onCancel} className="cancel-btn">Cancel</button>
        </div>
      </div>
    </div>
  );
};

// Section Editor Component
const SectionEditor = ({ onSave, onCancel }) => {
  const [sectionData, setSectionData] = useState({
    name: '',
    weight: 0,
    calculations: []
  });

  const handleSave = () => {
    if (!sectionData.name || sectionData.weight <= 0) {
      alert('Please fill in all required fields');
      return;
    }
    onSave(sectionData);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>Add New Section</h3>
        
        <div className="form-group">
          <label>Section Name:</label>
          <input
            type="text"
            value={sectionData.name}
            onChange={(e) => setSectionData(prev => ({ ...prev, name: e.target.value }))}
          />
        </div>

        <div className="form-group">
          <label>Section Weight (%):</label>
          <input
            type="number"
            value={sectionData.weight}
            onChange={(e) => setSectionData(prev => ({ ...prev, weight: parseInt(e.target.value) }))}
          />
        </div>

        <div className="modal-buttons">
          <button onClick={handleSave} className="save-btn">Save</button>
          <button onClick={onCancel} className="cancel-btn">Cancel</button>
        </div>
      </div>
    </div>
  );
};

export default Configure;
