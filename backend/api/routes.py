from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from extraction.extractor import extract
import tempfile
import os
import shutil
import json
import anthropic
import datetime
from pathlib import Path

router = APIRouter()

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

async def trigger_calculate_endpoint(calculations_json: dict, extracted_data: dict, txt_file_path: str) -> dict:
    """
    Trigger the calculate endpoint with the calculations JSON and extracted data.
    """
    try:
        # Get the API key from environment
        
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise Exception("ANTHROPIC_API_KEY not found in environment variables")
        
        # Initialize Anthropic client
        anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Read the generated text file content
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        
        # Prepare the prompt for Claude
        prompt = (
            "You are a data scientist, please match the values of the variables in the extracted file "
            "into the formulas found inside the json file to calculate the actual result of each of these formulas. "
            "Please return the data as a list of dictionaries: [{'formula': 'result'}, {'formula2': 'result2'}, ...]. "
            "Make sure to return valid JSON format.\n\n"
            f"Extracted Data from Text File:\n{txt_content}\n\n"
            f"Additional Extracted Data:\n{json.dumps(extracted_data, indent=2)}\n\n"
            f"Calculations JSON:\n{json.dumps(calculations_json, indent=2)}"
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
            "txt_file_used": txt_file_path,
            "message": "Calculations completed successfully using Claude Sonnet 3.5 with generated text file"
        }
        
    except Exception as e:
        print(f"Error in calculate endpoint: {e}")
        raise


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    safe_name = os.path.basename(file.filename).replace(" ", "_")
    temp_path = os.path.join(temp_dir, f"temp_{safe_name}")

    print(f"Saving to {temp_path}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract data from the uploaded file
        extracted_result = extract(temp_path)
        
        # Generate text file after extraction
        txt_file_path = await generate_text_file_after_extraction(extracted_result)
        
        # Load calculations JSON from frontend data
        calculations_json_path = Path(__file__).parent.parent.parent / "frontend" / "src" / "data" / "calculations.json"
        
        with open(calculations_json_path, 'r', encoding='utf-8') as f:
            calculations_json = json.load(f)
        
        # Automatically trigger the calculate endpoint
        try:
            calculation_result = await trigger_calculate_endpoint(calculations_json, extracted_result, txt_file_path)
            
            # Return both extraction and calculation results
            return JSONResponse(content={
                "success": True,
                "extracted_data": extracted_result,
                "calculation_result": calculation_result,
                "generated_txt_file": txt_file_path,
                "message": "File processed, text file generated, and credit score calculated successfully"
            })
            
        except Exception as calc_error:
            print(f"Calculation error: {calc_error}")
            # Return extraction result even if calculation fails
            return JSONResponse(content={
                "success": True,
                "extracted_data": extracted_result,
                "calculation_result": None,
                "generated_txt_file": txt_file_path,
                "calculation_error": str(calc_error),
                "message": "File extracted and text file generated successfully, but calculation failed"
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
