import json

json_str = '[{"generated_text": "Transform imperative to declarative: Write a function to split a string into parts.\\nWrite a function to split a string into parts.\\nWrite a function to split a string into parts. The function should take a string and return an array of strings.\\nThe function should return an array of strings.\\nThe function should"}]'
result = json.loads(json_str)

print(result[0]['generated_text'].split("\n"))

def find_declarative_text(imperative, declarative):
    lines = declarative.split("\n")
    # iterate from second line to avoid the prompt
    for line in lines[1:]:
        if line.strip():  # Check if the line is not empty
            # return line if it doesnot start with imperative
            if not line.strip().startswith(imperative):
                return line.strip()
            
    return None  # Return None if no suitable line is found

print(find_declarative_text('Write a function to split a string into parts', result[0]['generated_text']))
