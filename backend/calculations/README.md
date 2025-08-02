# Credit Scoring Calculation System

This document describes the new calculation endpoints and functionality for the Simah Credit Scorer application.

## Overview

The calculation system provides endpoints for:
1. **Formula Management** - Update and manage calculation formulas
2. **Credit Score Calculation** - Process extracted data through configured formulas
3. **Automatic Processing** - Upload endpoint now automatically triggers calculations

## New Endpoints

### 1. Update Formula Configuration
**Endpoint:** `PUT /api/calculations/formula`

Updates the calculations configuration file with new formulas and sections.

**Request Body:**
```json
{
  "sections": [
    {
      "name": "Traditional Score",
      "weight": 60,
      "calculations": [
        {
          "name": "Simah Score",
          "formula": "({credit_score} / 900) * 200",
          "variables": ["credit_score"],
          "weight": 100,
          "max_points": 200
        }
      ]
    }
  ],
  "variables": {
    "payment": ["Active Defaulted Products", "..."],
    "balance": ["Amount Applied For", "..."]
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Calculations configuration updated successfully",
  "sections_count": 2,
  "total_formulas": 3
}
```

### 2. Calculate Credit Score
**Endpoint:** `POST /api/calculations/calculate`

Processes extracted data through the configured formulas to generate credit scores.

**Request Body:**
```json
{
  "extracted_data": {
    "credit_score": "700",
    "monthly_income": "15000",
    "employment_duration_months": "36"
  }
}
```

**Response:**
```json
{
  "success": true,
  "calculation_summary": {
    "final_credit_score": 687.5,
    "score_percentage": 76.39,
    "total_sections": 2,
    "total_formulas": 3
  },
  "sections": [
    {
      "name": "Traditional Score",
      "weight": 60,
      "calculations": [
        {
          "name": "Simah Score",
          "formula": "({credit_score} / 900) * 200",
          "score": 155.56,
          "max_points": 200,
          "variables_used": {"credit_score": 700},
          "missing_variables": []
        }
      ],
      "total_score": 155.56,
      "weighted_score": 93.33
    }
  ],
  "final_result": {
    "final_credit_score": 687.5,
    "score_percentage": 76.39
  }
}
```

### 3. Get Current Configuration
**Endpoint:** `GET /api/calculations/formula`

Retrieves the current calculations configuration.

### 4. Validate Configuration
**Endpoint:** `GET /api/calculations/formula/validate`

Validates the current formula configuration for errors.

### 5. Health Check
**Endpoint:** `GET /api/calculations/health`

Health check for the calculations service.

## Updated Upload Endpoint

The `/upload` endpoint now automatically triggers credit score calculation:

**Response Format:**
```json
{
  "success": true,
  "extracted_data": { /* extracted fields */ },
  "calculation_result": { /* calculation results */ },
  "message": "File processed and credit score calculated successfully"
}
```

## Formula Syntax

### Variables
- Variables are enclosed in curly braces: `{variable_name}`
- Variable names are normalized (lowercase, underscores for spaces)

### Supported Functions
- `IF(condition, true_value, false_value)` - Conditional logic
- `MIN(value1, value2, ...)` - Minimum value
- `MAX(value1, value2, ...)` - Maximum value

### Operators
- `+`, `-`, `*`, `/` - Basic arithmetic
- `>=`, `<=`, `>`, `<`, `==`, `!=` - Comparison operators

### Example Formulas
```javascript
// Simple calculation
"({credit_score} / 900) * 200"

// Conditional logic
"IF({monthly_income} >= 20000, 150, IF({monthly_income} >= 15000, 120, 60))"

// With functions
"MIN({employment_duration_months} / 24, 1) * 80"
```

## Variable Matching

The system uses intelligent variable matching:

1. **Exact Match** - Direct variable name match
2. **Normalized Match** - After converting to lowercase and replacing spaces
3. **Fuzzy Match** - Partial string matching
4. **Semantic Match** - Predefined variations for common credit variables

### Common Variable Mappings
- `credit_score` → `credit_score`, `score`, `simah_score`
- `monthly_income` → `monthly_income`, `income`, `salary`, `total_salary`
- `employment_duration_months` → `employment_duration`, `employment_months`

## Error Handling

- **Missing Variables** - Defaults to 0, tracked in results
- **Formula Errors** - Returns 0 score, logs error
- **Configuration Errors** - Validation endpoint identifies issues

## Frontend Integration

### Configure.js Updates
- Uses `/api/calculations/formula` endpoint for saving
- Improved error handling and user feedback

### Results.js Updates
- Displays section-based scoring breakdown
- Shows formula details and variable usage
- Handles both new and legacy result formats
- Shows calculation errors and data analysis

## File Structure

```
backend/
├── calculations/
│   ├── __init__.py
│   ├── engine.py         # Core calculation logic
│   └── routes.py         # API endpoints
├── api/
│   └── routes.py         # Updated main routes
└── test_calculations.py  # Test file
```

## Testing

Run the test file to validate the calculation engine:

```bash
cd backend
python test_calculations.py
```

## Data Science Approach

The calculation engine implements data science principles:

1. **Variable Normalization** - Standardizes variable names for matching
2. **Fuzzy Matching** - Handles variations in extracted data field names
3. **Safe Evaluation** - Secure formula execution with error handling
4. **Weighted Scoring** - Proper mathematical weighting of section scores
5. **Comprehensive Reporting** - Detailed breakdown of calculation process

This system provides a robust, flexible foundation for credit scoring that can adapt to various data sources and scoring methodologies.
