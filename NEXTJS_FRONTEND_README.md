# VFIT Deep Researcher - Next.js Frontend

This document describes how to use the Next.js frontend for the VFIT Deep Researcher application.

## Overview

The VFIT Deep Researcher now includes a modern, responsive Next.js frontend that provides a ChatGPT-like interface to interact with the research tools. The frontend communicates with the Flask backend via API calls and Socket.IO for real-time updates.

## Prerequisites

- Node.js (v16+)
- npm or yarn
- Python 3.8+ (for backend)

## Running the Application

### Option 1: Full Stack with Single Command

Use the provided run script to start both the backend and frontend:

```bash
./run_fullstack.sh
```

This script will:
1. Start the Flask backend on port 5000
2. Start the Next.js frontend on port 3000
3. Handle cleanup when you press Ctrl+C

### Option 2: Run Components Separately

#### Start the Backend

```bash
python app.py
```

The Flask backend will run on http://localhost:5000

#### Start the Frontend

```bash
cd frontend/next-ui
npm run dev
```

The Next.js frontend will run on http://localhost:3000

## Features

The Next.js frontend provides:

- ChatGPT-like interface with conversation history
- Real-time status updates during research and quiz processes
- Support for markdown rendering in messages
- Mobile-responsive design
- Command shortcuts for common tasks

## Command Reference

The chat interface supports the following commands:

- `/research [topic]` - Start deep research on a topic
  - Example: `/research How do LSTM networks handle long sequences?`
  
- `/quiz [questions]` - Answer quiz questions using the knowledge base
  - Example: `/quiz What are the key components of a transformer architecture?`
  
- `/help` - Show available commands

## Architecture

The frontend is built with:

- Next.js 13 (App Router)
- TypeScript
- Tailwind CSS
- Socket.IO for real-time updates
- React Markdown for rendering markdown content

Communication with the backend is handled through:
- RESTful API calls for starting research and quiz processes
- Socket.IO for real-time status updates and results

## Troubleshooting

- If you encounter connection issues, make sure both the backend and frontend are running.
- Check the browser console for error messages.
- Make sure your .env.local file in the Next.js directory is properly configured.

## License

This project is proprietary and confidential. 