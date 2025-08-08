from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from extraction.extractor import extract
import tempfile
import os
import shutil
import json
import anthropic
import datetime
import re
from pathlib import Path
from typing import List, Dict, Any

router = APIRouter()

def parse_llm_response(llm_output: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM response to extract the JSON array of calculation results.
    Handles various formats that Claude might return.
    """
    try:
        # First, try to parse the entire response as JSON
        parsed = json.loads(llm_output.strip())
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            return []
    except json.JSONDecodeError:
        pass
    
    # If direct parsing fails, try to extract JSON from markdown code blocks
    # Look for JSON code blocks (```json ... ```)
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            continue
    
    # Look for any JSON array pattern in the text
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    array_matches = re.findall(array_pattern, llm_output, re.DOTALL)
    
    for match in array_matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, list):
                return parsed
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
    
    return results if results else []

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.get("/test-results")
def test_results():
    """Test endpoint to return sample LLM calculation results for frontend testing."""
    return {
        "success": True,
        "extracted_data": {
            "credit_score": 750,
            "monthly_income": 15000,
            "employment_duration_months": 36
        },
        "calculation_result": {
            "success": True,
            "results": [
                {"Simah Score": "166.67"},
                {"Income Level": "120"},
                {"Employment Stability": "80"}
            ],
            "raw_llm_output": '```json\n[\n    {"Simah Score": "166.67"},\n    {"Income Level": "120"},\n    {"Employment Stability": "80"}\n]\n```',
            "txt_file_used": "/path/to/file.txt",
            "message": "Calculations completed successfully using Claude Sonnet 3.5 with generated text file"
        },
        "generated_txt_file": "/path/to/extracted_data.txt",
        "message": "File processed, text file generated, and credit score calculated successfully"
    }

# load environment variables
from dotenv import load_dotenv
load_dotenv()


async def generate_text_file_after_extraction(extracted_result: dict) -> str:
    """
    Generate a text file with the extracted data after processing.
    Returns the path to the generated text file.
    """
    try:
        # Create the output directory if it doesn't exist
        output_dir = Path(__file__).parent.parent / "extracted_tables_enhanced"
        output_dir.mkdir(exist_ok=True)
        
        # Generate a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_file_path = output_dir / f"extracted_data_{timestamp}.txt"
        
        # Format the extracted data as text
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EXTRACTED DATA FROM UPLOADED FILE\n")
            f.write("=" * 80 + "\n\n")
            
            # Write the extracted data in a readable format
            if isinstance(extracted_result, dict):
                for key, value in extracted_result.items():
                    f.write(f"{key.upper()}:\n")
                    f.write("-" * 40 + "\n")
                    if isinstance(value, (list, dict)):
                        f.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n\n")
                    else:
                        f.write(str(value) + "\n\n")
            else:
                f.write(json.dumps(extracted_result, indent=2, ensure_ascii=False))
        
        print(f"Generated text file: {txt_file_path}")
        return str(txt_file_path)
        
    except Exception as e:
        print(f"Error generating text file: {e}")
        raise

async def trigger_calculate_endpoint(calculations_json: dict, txt_file_path: str) -> dict:
    """
    Trigger the calculate endpoint with the calculations JSON and extracted table data from file.
    """
    try:
        print("Starting LLM calculation...")
        
        # Read the extracted tables file content
        print(f"Reading tables file: {txt_file_path}")
        
        # Check if file exists
        if not Path(txt_file_path).exists():
            raise Exception(f"Tables file not found: {txt_file_path}")
            
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        print(f"Tables file content length: {len(txt_content)} characters")
        
        # Parse the text content to extract data
        # This is a simple approach - you might need to adjust based on your text format
        extracted_data = {}
        lines = txt_content.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Try to convert to number if possible
                try:
                    if '.' in value:
                        extracted_data[key] = float(value)
                    else:
                        extracted_data[key] = int(value)
                except ValueError:
                    extracted_data[key] = value
        
        print(f"Extracted data: {extracted_data}")
        
        # Now call the actual calculate_with_llm function directly
        from calculations.routes import calculate_with_llm, CalculateLLMRequest
        
        print("Calling calculate_with_llm function directly...")
        
        # Create the request object
        request = CalculateLLMRequest(
            calculations_json=calculations_json,
            extracted_data=extracted_data
        )
        
        # Call the function directly
        result = await calculate_with_llm(request)
        print(f"Calculate function response: {result}")
        return result
        
    except Exception as e:
        print(f"Error in trigger_calculate_endpoint: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise


# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     temp_dir = tempfile.gettempdir()
#     safe_name = os.path.basename(file.filename).replace(" ", "_")
#     temp_path = os.path.join(temp_dir, f"temp_{safe_name}")

#     print(f"Saving to {temp_path}")
#     with open(temp_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     try:
#         result = extract(temp_path)
#     finally:
#         os.remove(temp_path)

#     return JSONResponse(content=result)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    safe_name = os.path.basename(file.filename).replace(" ", "")  # Remove all spaces like main branch
    temp_path = os.path.join(temp_dir, f"temp{safe_name}")       # No underscore like main branch

    print(f"Saving to {temp_path}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract data from the uploaded file
        print("Starting extraction process...")
        extracted_result = extract(temp_path)
        print(f"Extraction completed. Result: {extracted_result}")
        
        # Load calculations JSON from backend data
        calculations_json_path = Path(__file__).parent.parent / "data" / "calculations.json"
        
        print(f"Loading calculations from: {calculations_json_path}")
        with open(calculations_json_path, 'r', encoding='utf-8') as f:
            calculations_json = json.load(f)
        print("Calculations JSON loaded successfully")
        
        # Automatically trigger the calculate endpoint
        try:
            print("Triggering calculation endpoint...")
            # Use the standard all_tables.txt file instead of generating a new one
            standard_tables_file = Path(__file__).parent.parent / "extracted_tables_enhanced" / "all_tables.txt"
            calculation_result = await trigger_calculate_endpoint(calculations_json, str(standard_tables_file))
            print(f"Calculation completed: {calculation_result}")
            
            # Return both extraction and calculation results
            return JSONResponse(content={
                "success": True,
                "extracted_data": extracted_result,
                "calculation_result": calculation_result,
                "tables_file_used": str(standard_tables_file),
                "message": "File processed and credit score calculated successfully using extracted tables"
            })
            
        except Exception as calc_error:
            print(f"Calculation error: {calc_error}")
            # Return extraction result even if calculation fails
            return JSONResponse(content={
                "success": True,
                "extracted_data": extracted_result,
                "calculation_result": None,
                "tables_file_used": str(standard_tables_file) if 'standard_tables_file' in locals() else "N/A",
                "calculation_error": str(calc_error),
                "message": "File extracted successfully, but calculation failed"
            })
            
    except Exception as extract_error:
        print(f"Extraction error: {extract_error}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": str(extract_error),
                "message": "Failed to extract data from file"
            }
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
