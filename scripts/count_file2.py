import os

def count_files_in_square_dirs(root_dir):
    count = 0
    for subdir, dirs, files in os.walk(root_dir):
        if 'square' in subdir:  # Check if we are inside a 'square' directory
            # Count files that end with '_ev-00.png'
            count += sum(1 for file in files if file.endswith('_ev-00.png'))
    return count

# Example usage:
root_directory = '../output/laion-aesthetics-1024'  # Replace with your directory path
file_count = count_files_in_square_dirs(root_directory)
print(f'Total files ending with "_ev-00.png": {file_count}')
