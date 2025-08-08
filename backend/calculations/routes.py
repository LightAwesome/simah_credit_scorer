from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from .config_handler import ConfigHandler
import anthropic
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()
config_handler = ConfigHandler()

# Debug: Check if environment variables are loaded
anthropic_api_key_debug = os.getenv("ANTHROPIC_API_KEY")
if anthropic_api_key_debug:
    print(f"ANTHROPIC_API_KEY found: {anthropic_api_key_debug[:10]}...")
else:
    print("ANTHROPIC_API_KEY NOT FOUND in environment variables")

def parse_llm_response(llm_output: str) -> Dict[str, Any]:
    """
    Parse the LLM response to extract the JSON array of calculation results.
    Handles various formats that Claude might return and structures the response
    with overall score and sections.
    """
    try:
        # First, try to parse the entire response as JSON
        parsed = json.loads(llm_output.strip())
        if isinstance(parsed, list):
            return _structure_parsed_results(parsed)
        elif isinstance(parsed, dict):
            return _structure_parsed_results([parsed])
        else:
            return {"overall_score": None, "sections": []}
    except json.JSONDecodeError:
        pass
    
    # If direct parsing fails, try to extract JSON from markdown code blocks
    import re
    
    # Look for JSON code blocks (```json ... ```)
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, list):
                return _structure_parsed_results(parsed)
            elif isinstance(parsed, dict):
                return _structure_parsed_results([parsed])
        except json.JSONDecodeError:
            continue
    
    # Look for any JSON array pattern in the text
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    array_matches = re.findall(array_pattern, llm_output, re.DOTALL)
    
    for match in array_matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, list):
                return _structure_parsed_results(parsed)
        except json.JSONDecodeError:
            continue
    
    # Look for individual JSON objects and combine them
    object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    object_matches = re.findall(object_pattern, llm_output)
    
    results = []
    for match in object_matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return _structure_parsed_results(results) if results else {"overall_score": None, "sections": []}

def _structure_parsed_results(parsed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Structure the parsed results into overall score and sections format.
    """
    overall_score = None
    sections = []
    
    for result in parsed_results:
        # Check if this is the overall score entry
        if "Overall Credit Score" in result:
            overall_score = result["Overall Credit Score"]
        else:
            # This is a section
            for section_name, section_data in result.items():
                if isinstance(section_data, dict):
                    section_total = None
                    formulas = []
                    
                    for key, value in section_data.items():
                        if key.startswith("total_"):
                            section_total = value
                        else:
                            formulas.append({"name": key, "value": value})
                    
                    sections.append({
                        "name": section_name,
                        "total": section_total,
                        "formulas": formulas
                    })
    
    return {
        "overall_score": overall_score,
        "sections": sections
    }

# Initialize Anthropic client with proper error handling
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    print("WARNING: ANTHROPIC_API_KEY not found in environment variables")
    anthropic_client = None
else:
    try:
        anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        print(f"Anthropic client initialized with API key: {anthropic_api_key[:10]}...")
    except Exception as e:
        print(f"ERROR initializing Anthropic client: {e}")
        anthropic_client = None

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
            print("ERROR: Extracted data is empty")
            raise HTTPException(status_code=400, detail="Extracted data cannot be empty")
        
        if not request.calculations_json:
            print("ERROR: Calculations JSON is empty")
            raise HTTPException(status_code=400, detail="Calculations JSON cannot be empty")
                
        # Get the API key from environment
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print("ERROR: ANTHROPIC_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found in environment variables")
                
        # Always create a fresh Anthropic client to ensure it has the correct API key
        try:
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            print("Anthropic client created successfully")
        except Exception as e:
            print(f"ERROR creating Anthropic client: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Anthropic client: {str(e)}")
        
        # Prepare the prompt for Claude
        prompt = (
            "You are a data scientist, please match the values of the variables in the extracted file "
            "into the formulas found inside the json file to calculate the actual result of each of these formulas. "
            "Please return the data as a list of dictionaries: [{'section1': {'total_section1': 'result1 + result2+...', 'formula1': 'result1', 'formula2': 'result2', ...}}, {'section2': {'total_section2': 'result1+result2+...', 'formula1': 'result1', 'formula2': 'result2', ...}}, ...]. "
            "Make sure to return valid JSON format.\n\n"
            f"Extracted Data:\n{json.dumps(request.extracted_data, indent=2)}\n\n"
            f"Calculations JSON:\n{json.dumps(request.calculations_json, indent=2)}"
            "Very important warning: Claude Sonnet 3.5 is very sensitive to the format of the input, no need to tell me if anything is missing, if we succeeded, just the below json of the list of dictionaries"
            "It is also very important to perform all the calculations in the formulas, do not skip any of them"
            "If the value of the calculation is not found or invalid variables, return 0"
            "DO NOT LEAVE OUT ANY OF THE FORMULAS!!!!This is a warning"
            ''' 
            ```json
[
            {'section1': 
                {'total_section1': 'result1 + result2+...', 
                'formula1': 'result1', 
                'formula2': 'result2', 
                ...},
            {'section2': 
                {'total_section2': 'result1+result2+...', 
                'formula1': 'result1', 
                'formula2': 'result2', 
                ...},
            ...
            },
            {'Overall Credit Score': 'sum of (all the sections) * (weight of the section) / 100'}
]
            
            ```
            '''
        )
        
        # Call Claude Sonnet 3.5
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
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
        
        
        # Parse the LLM response to extract JSON
        parsed_results = parse_llm_response(llm_output)
        
        return {
            "success": True,
            "overall_score": parsed_results.get("overall_score"),
            "sections": parsed_results.get("sections", []),
            "raw_llm_output": llm_output,
            "message": "Calculations completed successfully using Claude Sonnet 3.5"
        }
        
    except Exception as e:
        print(f"=== ERROR IN CALCULATE ENDPOINT: {str(e)} ===")
        import traceback
        print(traceback.format_exc())
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

@router.get("/variables")
async def get_variables():
    """
    Retrieve the current variables configuration.
    Returns all available variables organized by categories.
    """
    try:
        variables = config_handler.load_variables()
        
        return {
            "success": True,
            "variables": variables
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Variables configuration file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.put("/variables")
async def update_variables(variables: Dict[str, list]):
    """
    Update the variables configuration.
    """
    try:
        success = config_handler.save_variables(variables)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save variables configuration")
        
        return {
            "success": True,
            "message": "Variables configuration updated successfully",
            "categories_count": len(variables),
            "total_variables": sum(len(vars_list) for vars_list in variables.values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
