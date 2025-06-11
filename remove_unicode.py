"""
Script to find and remove Unicode characters from Python files.
This will help ensure compatibility with Windows cp1252 encoding.
"""
import os
import re
import codecs

def remove_unicode_from_file(filepath):
    """Remove Unicode characters from a single file."""
    try:
        # Read file with UTF-8 encoding
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Dictionary of Unicode replacements
        replacements = {
            '[WARNING]': '[WARNING]',
            '[ERROR]': '[ERROR]',
            '[OK]': '[OK]',
            '[SCREENSHOT]': '[SCREENSHOT]',
            '[GAME]': '[GAME]',
            '[TIMER]': '[TIMER]',
            '[START]': '[START]',
            '[COMPLETE]': '[COMPLETE]',
            '[SUMMARY]': '[SUMMARY]',
            '[DEBUG]': '[DEBUG]',
            '[DIR]': '[DIR]',
            '[CLEANUP]': '[CLEANUP]',
            '[SEARCH]': '[SEARCH]',
            '[CRASH]': '[CRASH]',
            '[FOUND]': '[FOUND]',
            '[SUCCESS]': '[SUCCESS]',
            '[TIP]': '[TIP]',
            '[STEP]': '[STEP]',
            '[WAIT]': '[WAIT]',
            '[PROGRESS]': '[PROGRESS]',
            '[DIALOG]': '[DIALOG]',
            '[EXIT]': '[EXIT]',
            '[INFO]': '[INFO]',
            '[ESC]': '[ESC]',
            '[SAVED]': '[SAVED]',
            '[TIMER]': '[TIMER]',
            '->': '->',
            '<-': '<-',
            '^': '^',
            'v': 'v',
            '*': '*',
            '*': '*',
            '-': '-',
            'o': 'o',
            '>': '>',
            '[OK]': '[OK]',
            '[X]': '[X]',
            '*': '*',
            '*': '*',
            '*': '*',
            '*': '*',
            '[#]': '[#]',
            '[]': '[]',
            '>': '>',
            '<': '<',
            '^': '^',
            'v': 'v',
        }
        
        # Replace Unicode characters
        modified = False
        for unicode_char, replacement in replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, replacement)
                modified = True
                print(f"  Replaced '{unicode_char}' with '{replacement}' in {filepath}")
        
        # Also remove any other non-ASCII characters
        original_content = content
        content = re.sub(r'[^\x00-\x7F]+', '', content)
        if content != original_content:
            modified = True
            print(f"  Removed non-ASCII characters from {filepath}")
        
        # Write back if modified
        if modified:
            with codecs.open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def scan_directory(directory='.'):
    """Scan directory for Python files with Unicode characters."""
    print(f"Scanning directory: {directory}")
    
    modified_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', 'venv', 'env', '.idea']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if remove_unicode_from_file(filepath):
                    modified_files.append(filepath)
    
    print(f"\nModified {len(modified_files)} files:")
    for file in modified_files:
        print(f"  - {file}")
    
    return modified_files

if __name__ == "__main__":
    import sys
    
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print("Unicode Character Remover")
    print("========================")
    print("This script will remove Unicode characters from Python files")
    print("and replace them with ASCII equivalents.")
    print()
    
    response = input(f"Scan directory '{directory}' for Unicode characters? (y/n): ")
    if response.lower() == 'y':
        scan_directory(directory)
        print("\nDone!")
    else:
        print("Cancelled.")
