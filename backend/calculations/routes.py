from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from .config_handler import ConfigHandler
import anthropic
import json
import os

router = APIRouter()
config_handler = ConfigHandler()

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")  # Make sure to set this environment variable
)

class FormulaUpdateRequest(BaseModel):
    """Request model for updating calculations configuration."""
    sections: List[Dict[str, Any]]
    variables: Optional[Dict[str, List[str]]] = None

class CalculateLLMRequest(BaseModel):
    """Request model for LLM-based calculations."""
    calculations_json: Dict[str, Any]
    extracted_data: Dict[str, Any]

@router.put("/formula")
async def update_formula_configuration(request: FormulaUpdateRequest):
    """
    Update the calculations configuration file.
    This endpoint allows updating the formulas, sections, and variables
    used for credit score calculations.
    """
    try:
        # Validate the request data
        if not request.sections:
            raise HTTPException(status_code=400, detail="Sections cannot be empty")
        
        # Construct the new configuration
        new_config = {
            "sections": request.sections
        }
        
        # Include variables if provided, otherwise keep existing
        if request.variables:
            new_config["variables"] = request.variables
        else:
            # Load current config to preserve variables
            try:
                current_config = config_handler.load_calculations_config()
                new_config["variables"] = current_config.get("variables", {})
            except:
                new_config["variables"] = {}
        
        # Validate sections structure
        for i, section in enumerate(request.sections):
            if not isinstance(section, dict):
                raise HTTPException(status_code=400, detail=f"Section {i} must be an object")
            
            required_fields = ["name", "weight", "calculations"]
            for field in required_fields:
                if field not in section:
                    raise HTTPException(status_code=400, detail=f"Section {i} missing required field: {field}")
            
            # Validate calculations within section
            for j, calc in enumerate(section["calculations"]):
                calc_required = ["name", "formula", "weight", "max_points"]
                for field in calc_required:
                    if field not in calc:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Calculation {j} in section '{section['name']}' missing required field: {field}"
                        )
        
        # Save the configuration
        success = config_handler.save_calculations_config(new_config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save calculations configuration")
        
        return {
            "success": True,
            "message": "Calculations configuration updated successfully",
            "sections_count": len(request.sections),
            "total_formulas": sum(len(section["calculations"]) for section in request.sections)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/calculate")
async def calculate_with_llm(request: CalculateLLMRequest):
    """
    Calculate credit score using Claude Sonnet 3.5.
    Accepts calculations JSON and extracted data, sends them to Claude,
    and returns Claude's output as a list of dictionaries.
    """
    try:
        if not request.extracted_data:
            raise HTTPException(status_code=400, detail="Extracted data cannot be empty")
        
        if not request.calculations_json:
            raise HTTPException(status_code=400, detail="Calculations JSON cannot be empty")
        
        # Prepare the prompt for Claude
        prompt = (
            "You are a data scientist, please match the values of the variables in the extracted file "
            "into the formulas found inside the json file to calculate the actual result of each of these formulas. "
            "Please return the data as a list of dictionaries: [{'formula': 'result'}, {'formula2': 'result2'}, ...]. "
            "Make sure to return valid JSON format.\n\n"
            f"Extracted Data:\n{json.dumps(request.extracted_data, indent=2)}\n\n"
            f"Calculations JSON:\n{json.dumps(request.calculations_json, indent=2)}"
        )
        
        # Call Claude Sonnet 3.5
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract the response
        llm_output = message.content[0].text
        
        return {
            "success": True,
            "results": llm_output,
            "message": "Calculations completed successfully using Claude Sonnet 3.5"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM calculation failed: {str(e)}")

@router.get("/formula")
async def get_current_formula_configuration():
    """
    Retrieve the current calculations configuration.
    Returns the formulas, sections, and variables currently in use.
    """
    try:
        config = config_handler.load_calculations_config()
        
        return {
            "success": True,
            "config": config
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Calculations configuration file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
