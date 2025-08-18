import json
from typing import Dict, Any
from pathlib import Path

class ConfigHandler:
    """
    Simple handler for loading and saving calculations configuration.
    Replaces the heavy CalculationEngine with just the config management needed.
    """
    
    def __init__(self):
        self.calculations_path = Path(__file__).parent.parent / "data" / "calculations.json"
        self.variables_path = Path(__file__).parent.parent / "data" / "variables.json"
        self.final_decision_path = Path(__file__).parent.parent / "data" / "Final_decision.json"
    
    def load_calculations_config(self) -> Dict[str, Any]:
        """Load the calculations configuration from JSON file."""
        try:
            with open(self.calculations_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Load variables from separate file and merge them
            variables = self.load_variables()
            config['variables'] = variables
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Calculations config not found at {self.calculations_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in calculations config: {e}")
    
    def load_variables(self) -> Dict[str, list]:
        """Load variables from the separate variables.json file."""
        try:
            with open(self.variables_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # If variables file doesn't exist, return empty dict
            return {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in variables config: {e}")
    
    def save_calculations_config(self, config: Dict[str, Any]) -> bool:
        """Save the calculations configuration to JSON file."""
        try:
            # Extract variables and save them separately
            variables = config.get('variables', {})
            config_without_variables = {k: v for k, v in config.items() if k != 'variables'}
            
            # Save main config
            with open(self.calculations_path, 'w', encoding='utf-8') as f:
                json.dump(config_without_variables, f, indent=2, ensure_ascii=False)
            
            # Save variables separately
            with open(self.variables_path, 'w', encoding='utf-8') as f:
                json.dump(variables, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving calculations config: {e}")
            return False
    
    def save_variables(self, variables: Dict[str, list]) -> bool:
        """Save variables to the separate variables.json file."""
        try:
            with open(self.variables_path, 'w', encoding='utf-8') as f:
                json.dump(variables, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving variables config: {e}")
            return False
    
    def load_final_decision_config(self) -> Dict[str, str]:
        """Load the final decision configuration from JSON file."""
        try:
            with open(self.final_decision_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Final decision config not found at {self.final_decision_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in final decision config: {e}")
    
    def save_final_decision_config(self, config: Dict[str, str]) -> bool:
        """Save the final decision configuration to JSON file."""
        try:
            with open(self.final_decision_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving final decision config: {e}")
            return False
