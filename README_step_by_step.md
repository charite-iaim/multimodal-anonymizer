# MIMIC-IV Anonymization Pipeline - Setup Guide

## Prerequisites

### Required Software

- **Python 3.13**
- **Node.js v24.13.0 and npm v11.6.2**
- **FFmpeg** (for audio processing)
- **Poppler** (for PDF processing)

### Installing Node.js with nvm

```bash
# Download and install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# Reload shell configuration (or restart your terminal)
. "$HOME/.nvm/nvm.sh"

# Download and install Node.js
nvm install 24

# Verify the Node.js version
node -v # Should print "v24.13.0"

# Verify npm version
npm -v # Should print "11.6.2"
```

### Installing System Dependencies (macOS)

```bash
brew install ffmpeg
brew install poppler
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mimiciv-anonymization-pipeline
```

load models from git lfs
brew install git-lfs

### 2. Configure Azure Credentials

Create a `.env` file in the project root with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
AZURE_DEPLOYMENT_NAME=your_azure_deployment_name
```

### 3. Set Up Python Virtual Environment

```bash
# Create virtual environment with Python 3.13
python3.13 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Running the Application

### Start the Backend Server

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Navigate to backend directory and start server
cd backend
python -m uvicorn app.main:app --reload
```

The backend will be available at `http://localhost:8000`

### Start the Frontend Development Server

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in terminal)

## Notes

- Keep both servers running during development
- The backend must be running for the frontend to function properly
- Make sure your `.env` file is properly configured before starting the servers