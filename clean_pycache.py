import os
import shutil

def remove_pycache(root_dir):
    """
    Recursively searches for and deletes __pycache__ directories within the given root directory.
    """
    deleted_count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_dir = os.path.join(dirpath, '__pycache__')
            print(f"Removing: {pycache_dir}")
            try:
                shutil.rmtree(pycache_dir)
                deleted_count += 1
            except Exception as e:
                print(f"Error removing {pycache_dir}: {e}")
    
    print(f"Finished cleaning. Removed {deleted_count} __pycache__ directories.")

if __name__ == "__main__":
    # Use the directory where the script is executed as the root directory
    current_dir = os.getcwd()
    print(f"Scanning for __pycache__ directories in: {current_dir}")
    remove_pycache(current_dir)