import sys
import os

# Get the directory of the current script
current_script_dir = os.path.dirname(__file__)

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Now, use an absolute import for the 'data' module within the 'lib' directory
from lib import data as anime_data
