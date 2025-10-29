import io from 'socket.io-client';

// Socket instance
let socket = null;

// Base URL for API
const BASE_URL = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5000';

// Initialize Socket.IO connection
export const initializeSocket = () => {
  if (!socket) {
    socket = io(BASE_URL, {
      reconnectionAttempts: 5,
      timeout: 10000,
      transports: ['websocket', 'polling']
    });

    socket.on('connect', () => {
      console.log('Socket connected');
    });

    socket.on('disconnect', () => {
      console.log('Socket disconnected');
    });

    socket.on('error', (error) => {
      console.error('Socket error:', error);
    });
  }
  return socket;
};

// Disconnect socket
export const disconnectSocket = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
};

// Get current processing status
export const getCurrentStatus = async () => {
  try {
    const response = await fetch(`${BASE_URL}/api/status`);
    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching status:', error);
    return { status: 'error', message: error.message };
  }
};

// Subscribe to status updates
export const subscribeToStatusUpdates = (callback) => {
  if (!socket) {
    initializeSocket();
  }

  socket.on('status_update', (data) => {
    callback(data);
  });

  return () => {
    socket.off('status_update');
  };
};

// Subscribe to research complete events
export const subscribeToResearchComplete = (callback) => {
  if (!socket) {
    initializeSocket();
  }

  socket.on('research_complete', (data) => {
    callback(data);
  });

  return () => {
    socket.off('research_complete');
  };
};

// Subscribe to research error events
export const subscribeToResearchError = (callback) => {
  if (!socket) {
    initializeSocket();
  }

  socket.on('research_error', (data) => {
    callback(data);
  });

  return () => {
    socket.off('research_error');
  };
};

// Subscribe to quiz complete events
export const subscribeToQuizComplete = (callback) => {
  if (!socket) {
    initializeSocket();
  }

  socket.on('quiz_complete', (data) => {
    callback(data);
  });

  return () => {
    socket.off('quiz_complete');
  };
};

// Subscribe to quiz error events
export const subscribeToQuizError = (callback) => {
  if (!socket) {
    initializeSocket();
  }

  socket.on('quiz_error', (data) => {
    callback(data);
  });

  return () => {
    socket.off('quiz_error');
  };
};

// Start research on a topic
export const startResearch = async (topic) => {
  try {
    const response = await fetch(`${BASE_URL}/api/research`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ topic }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || `Error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error starting research:', error);
    throw error;
  }
};

// Answer a quiz
export const answerQuiz = async (quizText) => {
  try {
    const response = await fetch(`${BASE_URL}/api/quiz`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ quiz_text: quizText }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `Error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error answering quiz:', error);
    throw error;
  }
};

// Export other functions as needed 