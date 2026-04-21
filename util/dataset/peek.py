import json
import sys

def peek_json_format(filepath, max_samples=5):
    """Quickly show JSON structure by sampling first few items"""
    
    def get_structure(obj, depth=0, max_depth=3):
        """Recursively get structure of JSON object"""
        if depth > max_depth:
            return "..."
        
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            result = {}
            for k, v in list(obj.items())[:10]:  # Limit keys shown
                result[k] = get_structure(v, depth + 1, max_depth)
            if len(obj) > 10:
                result[f"... and {len(obj)-10} more keys"] = "..."
            return result
        
        elif isinstance(obj, list):
            if not obj:
                return "[]"
            # Sample first few items
            samples = [get_structure(item, depth + 1, max_depth) 
                      for item in obj[:3]]
            if len(set(str(s) for s in samples)) > 1:
                return f"List[Union[{', '.join(str(s) for s in samples)}]]"
            return f"List[{samples[0] if samples else '?'}]"
        
        elif isinstance(obj, str):
            return "string"
        elif isinstance(obj, bool):
            return "boolean"
        elif isinstance(obj, int):
            return "integer"
        elif isinstance(obj, float):
            return "float"
        elif obj is None:
            return "null"
        else:
            return type(obj).__name__
    
    try:
        with open(filepath, 'r') as f:
            # Try to parse as streaming for huge arrays
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # It's an array, sample first few items
                data = json.loads(f.read())
                if isinstance(data, list):
                    print(f"Root is an array with {len(data)} items")
                    print("\nStructure of first item:")
                    print(json.dumps(get_structure(data[0]), indent=2))
                    if len(data) > 1:
                        print("\nStructure of second item (to check consistency):")
                        print(json.dumps(get_structure(data[1]), indent=2))
                else:
                    print(json.dumps(get_structure(data), indent=2))
            else:
                # Regular JSON object
                data = json.load(f)
                print(json.dumps(get_structure(data), indent=2))
                
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        peek_json_format(sys.argv[1])
    else:
        print("Usage: python peek_json.py <filename>")