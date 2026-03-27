import os
import re
from pathlib import Path

def rename_pdfs(folder_path):
    """
    Renames PDF files from format '(YY, MM.) Name.pdf' to '(YYYY.MM) Name.pdf'
    
    Args:
        folder_path: Path to the folder containing PDF files
    """
    # Pattern to match files like "(16, 07.) Name.pdf"
    pattern = r'^\((\d{2}),\s*(\d{2})\.\)\s*(.+\.pdf)$'
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    renamed_count = 0
    skipped_count = 0
    
    # Get all PDF files in the folder
    pdf_files = list(folder.glob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) in '{folder_path}'")
    print("-" * 60)
    
    for file_path in pdf_files:
        filename = file_path.name
        match = re.match(pattern, filename)
        
        if match:
            year_short = match.group(1)
            month = match.group(2)
            rest_of_name = match.group(3)
            
            # Convert 2-digit year to 4-digit year (assuming 2000s)
            year_full = f"20{year_short}"
            
            # Create new filename
            new_filename = f"({year_full}.{month}) {rest_of_name}"
            new_file_path = file_path.parent / new_filename
            
            # Check if target file already exists
            if new_file_path.exists():
                print(f"SKIP: '{filename}' -> Target file already exists")
                skipped_count += 1
            else:
                # Rename the file
                file_path.rename(new_file_path)
                print(f"✓ '{filename}' -> '{new_filename}'")
                renamed_count += 1
        else:
            print(f"SKIP: '{filename}' -> Does not match expected format")
            skipped_count += 1
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Renamed: {renamed_count} file(s)")
    print(f"  Skipped: {skipped_count} file(s)")

if __name__ == "__main__":
    # You can change this to your folder path
    folder_path = input("Enter the folder path containing PDF files: ").strip()
    
    # Remove quotes if user wraps path in quotes
    folder_path = folder_path.strip('"').strip("'")
    
    rename_pdfs(folder_path)