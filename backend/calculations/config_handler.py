import json
from typing import Dict, Any
from pathlib import Path

class ConfigHandler:
    """
    Simple handler for loading and saving calculations configuration.
    Replaces the heavy CalculationEngine with just the config management needed.
    """
    
    def __init__(self):
        self.calculations_path = Path(__file__).parent.parent.parent / "frontend" / "src" / "data" / "calculations.json"
    
    def load_calculations_config(self) -> Dict[str, Any]:
        """Load the calculations configuration from JSON file."""
        try:
            with open(self.calculations_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calculations config not found at {self.calculations_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in calculations config: {e}")
    
    def save_calculations_config(self, config: Dict[str, Any]) -> bool:
        """Save the calculations configuration to JSON file."""
        try:
            with open(self.calculations_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving calculations config: {e}")
            return False
