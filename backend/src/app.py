import os, sys
# Ensure project root is in path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import the FastAPI app defined in the top-level app.py
from app import app as fastapi_app

# Export as variable expected by uvicorn
app = fastapi_app
