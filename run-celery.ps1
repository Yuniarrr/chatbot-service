Set-Location "F:\project\chatbot-ta\chatbot-service"

# Activate the virtual environment
& ".\.venv\Scripts\Activate.ps1"

# Start the Celery worker using the 'solo' pool (Windows-compatible)
celery -A app.task worker --loglevel=info
