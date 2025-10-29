import React, { useState, useEffect, useRef } from 'react';
import SendIcon from '@mui/icons-material/Send';
import AddIcon from '@mui/icons-material/Add';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import DeleteIcon from '@mui/icons-material/Delete';
import HistoryIcon from '@mui/icons-material/History';
import SearchIcon from '@mui/icons-material/Search';
import QuizIcon from '@mui/icons-material/Quiz';
import HelpIcon from '@mui/icons-material/Help';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';
import UpdateIcon from '@mui/icons-material/Update';
import ReactMarkdown from 'react-markdown';

// API services for connecting to backend
import { 
  getCurrentStatus,
  initializeSocket,
  disconnectSocket,
  subscribeToStatusUpdates,
  subscribeToResearchComplete,
  subscribeToResearchError,
  subscribeToQuizComplete,
  subscribeToQuizError,
  startResearch,
  answerQuiz,
} from '../services/apiService';

function ChatPageTailwind() {
  const [input, setInput] = useState('');
  const [conversations, setConversations] = useState([
    { id: 1, title: "Previous Research", updated: "2 hours ago" },
    { id: 2, title: "MTV1797 Analysis", updated: "Yesterday" },
    { id: 3, title: "VIT-123 Investigation", updated: "3 days ago" },
  ]);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [processingStatus, setProcessingStatus] = useState({});
  const messagesEndRef = useRef(null);

  // Initialize socket connection for real-time updates
  useEffect(() => {
    initializeSocket();
    
    // Subscribe to various status updates
    const unsubscribeStatus = subscribeToStatusUpdates((data) => {
      setProcessingStatus(data);
      
      // Update UI based on processing status
      if (data.status === 'completed' || data.status === 'error') {
        setLoading(false);
      }
    });
    
    // Subscribe to research complete updates
    const unsubscribeResearchComplete = subscribeToResearchComplete((data) => {
      handleResearchComplete(data);
    });
    
    // Subscribe to research error updates
    const unsubscribeResearchError = subscribeToResearchError((data) => {
      handleError('Research error', data.error);
    });
    
    // Subscribe to quiz complete updates
    const unsubscribeQuizComplete = subscribeToQuizComplete((data) => {
      handleQuizComplete(data);
    });
    
    // Subscribe to quiz error updates
    const unsubscribeQuizError = subscribeToQuizError((data) => {
      handleError('Quiz error', data.error);
    });
    
    // Clean up on unmount
    return () => {
      unsubscribeStatus();
      unsubscribeResearchComplete();
      unsubscribeResearchError();
      unsubscribeQuizComplete();
      unsubscribeQuizError();
      disconnectSocket();
    };
  }, []);

  // Initialize with welcome message and command help when a conversation is selected
  useEffect(() => {
    if (selectedConversation) {
      setMessages([
        { 
          id: 1, 
          role: 'system', 
          content: 'Welcome to VFIT Deep Research Chat. I can help you with research and quiz answering. Try these commands:\n\n' +
                  '• `/research [topic]` - Start deep research on a topic\n' +
                  '• `/quiz [questions]` - Answer a quiz or questions\n' +
                  '• `/help` - Show available commands\n\n' +
                  'Or just ask me anything about enterprise data, documentation, or code!'
        },
      ]);
      scrollToBottom();
    }
  }, [selectedConversation]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle processing status updates
  useEffect(() => {
    // If there's a status update and we're loading, add it as a system message
    if (processingStatus.current_step && loading) {
      // Check if we already have a status message
      const statusMessageIndex = messages.findIndex(m => m.isStatusMessage);
      
      const statusContent = `**Current Status**: ${processingStatus.status}\n**Step**: ${processingStatus.current_step}\n**Progress**: ${processingStatus.progress}%\n\n${processingStatus.log_messages ? "**Latest logs**:\n" + processingStatus.log_messages.slice(-3).join("\n") : ""}`;
      
      if (statusMessageIndex >= 0) {
        // Update existing status message
        const updatedMessages = [...messages];
        updatedMessages[statusMessageIndex] = {
          ...updatedMessages[statusMessageIndex],
          content: statusContent
        };
        setMessages(updatedMessages);
      } else {
        // Add new status message
        setMessages(prev => [...prev, {
          id: Date.now(),
          role: 'system',
          content: statusContent,
          isStatusMessage: true
        }]);
      }
    }
  }, [processingStatus, loading, messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleNewChat = () => {
    const newId = conversations.length > 0 ? Math.max(...conversations.map(c => c.id)) + 1 : 1;
    const newConversation = { id: newId, title: "New Conversation", updated: "Just now" };
    setConversations([newConversation, ...conversations]);
    setSelectedConversation(newConversation);
    setMessages([
      { 
        id: 1, 
        role: 'system', 
        content: 'Welcome to VFIT Deep Research Chat. I can help you with research and quiz answering. Try these commands:\n\n' +
                '• `/research [topic]` - Start deep research on a topic\n' +
                '• `/quiz [questions]` - Answer a quiz or questions\n' +
                '• `/help` - Show available commands\n\n' +
                'Or just ask me anything about enterprise data, documentation, or code!'
      },
    ]);
  };

  // Handle research completion
  const handleResearchComplete = (data) => {
    const summary = data.summary || "No summary available.";
    const assistantMessage = { 
      id: Date.now(), 
      role: 'assistant', 
      content: `# Research Complete\n\n${summary}\n\n[Download Full Report](${data.result_file})`
    };
    
    // Remove any status message first
    const filteredMessages = messages.filter(m => !m.isStatusMessage);
    setMessages([...filteredMessages, assistantMessage]);
    setLoading(false);
  };

  // Handle quiz completion
  const handleQuizComplete = (data) => {
    const answers = data.answers || [];
    let content = "# Quiz Answers\n\n";
    
    if (answers.length > 0) {
      answers.forEach((answer, index) => {
        content += `## Question ${index + 1}: ${answer.question}\n\n`;
        content += `**Answer**: ${answer.answer}\n\n`;
        content += `**Justification**: ${answer.justification}\n\n`;
        if (index < answers.length - 1) content += "---\n\n";
      });
    } else {
      content += "No answers were generated. Please try again with a different quiz.";
    }
    
    const assistantMessage = { 
      id: Date.now(), 
      role: 'assistant', 
      content: content
    };
    
    // Remove any status message first
    const filteredMessages = messages.filter(m => !m.isStatusMessage);
    setMessages([...filteredMessages, assistantMessage]);
    setLoading(false);
  };

  // Handle API errors
  const handleError = (type, error) => {
    const errorMessage = { 
      id: Date.now(), 
      role: 'system', 
      content: `**Error**: ${type} - ${error}\n\nPlease try again or use a different approach.`,
      isError: true
    };
    
    // Remove any status message first
    const filteredMessages = messages.filter(m => !m.isStatusMessage);
    setMessages([...filteredMessages, errorMessage]);
    setLoading(false);
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    
    // Add user message
    const userMessage = { id: Date.now(), role: 'user', content: input };
    setMessages([...messages, userMessage]);
    setInput('');
    
    // Check if the input is a command
    const trimmedInput = input.trim();
    if (trimmedInput.startsWith('/')) {
      await handleCommand(trimmedInput);
    } else {
      setLoading(true);
      
      // Default chat behavior - simulate assistant response after delay
      setTimeout(() => {
        const assistantMessage = { 
          id: Date.now(), 
          role: 'assistant', 
          content: `I'm analyzing your question about "${input.substring(0, 20)}..."\n\n` +
                   `Here are my findings based on enterprise data:\n\n` +
                   `1. **JIRA Analysis**: Several tickets related to this topic were found, including VIT-456 and MTV1234.\n` +
                   `2. **Confluence Documentation**: Technical specifications are available in the [System Architecture](https://example.com) page.\n` +
                   `3. **Code Analysis**: Recent changes in CL 12345 show implementation details.\n\n` +
                   `Would you like me to analyze anything specific about this topic?`
        };
        setMessages(prevMessages => [...prevMessages, assistantMessage]);
        setLoading(false);
        
        // Update conversation title if it's a new conversation
        if (selectedConversation.title === "New Conversation") {
          const truncatedTitle = input.length > 20 ? `${input.substring(0, 20)}...` : input;
          setConversations(prevConversations => 
            prevConversations.map(conv => 
              conv.id === selectedConversation.id 
                ? { ...conv, title: truncatedTitle, updated: "Just now" } 
                : conv
            )
          );
          setSelectedConversation(prev => ({ ...prev, title: truncatedTitle }));
        }
      }, 2000);
    }
  };

  // Handle special commands
  const handleCommand = async (command) => {
    const parts = command.split(' ');
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1).join(' ');
    
    setLoading(true);
    
    try {
      switch (cmd) {
        case '/research':
          if (!args) {
            setMessages(prev => [...prev, { 
              id: Date.now(), 
              role: 'system', 
              content: 'Please provide a research topic after the /research command.',
              isError: true 
            }]);
            setLoading(false);
            return;
          }
          
          const researchResponse = await startResearch(args);
          
          // Add a processing message
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            role: 'system', 
            content: `Starting research on "${args}"...\nThis may take a few minutes.\n\nStatus: initializing`,
            isStatusMessage: true
          }]);
          
          // Update conversation title
          const researchTitle = args.length > 20 ? `${args.substring(0, 20)}...` : args;
          setConversations(prevConversations => 
            prevConversations.map(conv => 
              conv.id === selectedConversation.id 
                ? { ...conv, title: `Research: ${researchTitle}`, updated: "Just now" } 
                : conv
            )
          );
          setSelectedConversation(prev => ({ ...prev, title: `Research: ${researchTitle}` }));
          break;
          
        case '/quiz':
          if (!args) {
            setMessages(prev => [...prev, { 
              id: Date.now(), 
              role: 'system', 
              content: 'Please provide quiz questions after the /quiz command.',
              isError: true 
            }]);
            setLoading(false);
            return;
          }
          
          const quizResponse = await answerQuiz(args);
          
          // Add a processing message
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            role: 'system', 
            content: `Analyzing quiz questions...\nThis may take a few minutes.\n\nStatus: initializing`,
            isStatusMessage: true
          }]);
          
          // Update conversation title
          const quizTitle = args.length > 20 ? `${args.substring(0, 20)}...` : args;
          setConversations(prevConversations => 
            prevConversations.map(conv => 
              conv.id === selectedConversation.id 
                ? { ...conv, title: `Quiz: ${quizTitle}`, updated: "Just now" } 
                : conv
            )
          );
          setSelectedConversation(prev => ({ ...prev, title: `Quiz: ${quizTitle}` }));
          break;
          
        case '/help':
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            role: 'system', 
            content: '# Available Commands\n\n' +
                    '• `/research [topic]` - Start deep research on a topic using JIRA, Confluence, and Perforce data\n' +
                    '• `/quiz [questions]` - Answer a quiz based on enterprise data\n' +
                    '• `/help` - Show this help message\n\n' +
                    'You can also just chat normally to ask questions about enterprise data!'
          }]);
          setLoading(false);
          break;
          
        default:
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            role: 'system', 
            content: `Unknown command: ${cmd}\nType /help to see available commands.`,
            isError: true
          }]);
          setLoading(false);
      }
    } catch (error) {
      handleError('Command error', error.message || 'Unknown error');
    }
  };

  const handleInputKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const renderMessageContent = (content) => {
    return (
      <ReactMarkdown>
        {content}
      </ReactMarkdown>
    );
  };
  
  const handleDeleteConversation = (e, conversationId) => {
    e.stopPropagation();
    
    // Filter out the deleted conversation
    const updatedConversations = conversations.filter(conv => conv.id !== conversationId);
    setConversations(updatedConversations);
    
    // If the deleted conversation was selected, select none
    if (selectedConversation?.id === conversationId) {
      setSelectedConversation(null);
    }
  };

  // Return empty state if no conversation is selected
  if (!selectedConversation) {
    return (
      <div className="flex flex-col h-[calc(100vh-120px)] max-w-7xl mx-auto px-2 py-3">
        <div className="flex h-full">
          {/* Sidebar */}
          <div className="w-1/4 border-r border-gray-700 h-full flex flex-col bg-gray-900 rounded-l-xl shadow-md overflow-hidden">
            <button 
              onClick={handleNewChat}
              className="m-2 mb-2 flex items-center justify-center bg-gradient-to-r from-blue-600 to-blue-500 text-white py-3 px-4 rounded-xl font-semibold hover:from-blue-700 hover:to-blue-600 shadow-lg hover:shadow-xl transition-all"
            >
              <AddIcon className="mr-2" />
              New Chat
            </button>
            
            <div className="px-2.5 py-1.5 font-semibold text-gray-400 flex items-center">
              <HistoryIcon fontSize="small" className="mr-1.5 opacity-70" />
              Previous Conversations
            </div>
            
            <div className="overflow-y-auto flex-1 px-1.5 py-1">
              {conversations.map((conversation) => (
                <div 
                  key={conversation.id}
                  onClick={() => setSelectedConversation(conversation)}
                  className="rounded-xl mb-1 py-1.8 pl-2 cursor-pointer transition-all duration-200 hover:bg-blue-800/10 hover:-translate-y-0.5"
                >
                  <div>
                    <div className="font-medium text-gray-200">{conversation.title}</div>
                    <div className="text-xs text-gray-400 flex justify-between pr-2">
                      <span>{conversation.updated}</span>
                      <button
                        onClick={(e) => handleDeleteConversation(e, conversation.id)}
                        className="text-gray-500 hover:text-red-400 transition-colors p-1"
                      >
                        <DeleteIcon fontSize="small" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Empty state - main content area */}
          <div className="w-3/4 flex flex-col items-center justify-center bg-gray-800 rounded-r-xl text-center p-8">
            <div className="mb-6">
              <SmartToyIcon style={{ fontSize: 60 }} className="text-blue-400 mb-4" />
              <h1 className="text-2xl font-bold text-white mb-2">VFIT Deep Research</h1>
              <p className="text-gray-400 max-w-lg">
                Select a conversation from the sidebar or start a new chat to begin interacting with the AI research assistant.
              </p>
            </div>
            
            <button
              onClick={handleNewChat}
              className="mt-4 flex items-center bg-blue-600 text-white py-2 px-6 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              <AddIcon className="mr-2" />
              Start New Chat
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Return conversation view if a conversation is selected
  return (
    <div className="flex flex-col h-[calc(100vh-120px)] max-w-7xl mx-auto px-2 py-3">
      <div className="flex h-full">
        {/* Sidebar */}
        <div className="w-1/4 border-r border-gray-700 h-full flex flex-col bg-gray-900 rounded-l-xl shadow-md overflow-hidden">
          <button 
            onClick={handleNewChat}
            className="m-2 mb-2 flex items-center justify-center bg-gradient-to-r from-blue-600 to-blue-500 text-white py-3 px-4 rounded-xl font-semibold hover:from-blue-700 hover:to-blue-600 shadow-lg hover:shadow-xl transition-all"
          >
            <AddIcon className="mr-2" />
            New Chat
          </button>
          
          <div className="px-2.5 py-1.5 font-semibold text-gray-400 flex items-center">
            <HistoryIcon fontSize="small" className="mr-1.5 opacity-70" />
            Previous Conversations
          </div>
          
          <div className="overflow-y-auto flex-1 px-1.5 py-1">
            {conversations.map((conversation) => (
              <div 
                key={conversation.id}
                onClick={() => setSelectedConversation(conversation)}
                className={`rounded-xl mb-1 py-1.8 pl-2 cursor-pointer transition-all duration-200 hover:bg-blue-800/10 hover:-translate-y-0.5 ${selectedConversation?.id === conversation.id ? 'bg-blue-900/20 border-l-2 border-blue-500' : ''}`}
              >
                <div>
                  <div className="font-medium text-gray-200">{conversation.title}</div>
                  <div className="text-xs text-gray-400 flex justify-between pr-2">
                    <span>{conversation.updated}</span>
                    <button
                      onClick={(e) => handleDeleteConversation(e, conversation.id)}
                      className="text-gray-500 hover:text-red-400 transition-colors p-1"
                    >
                      <DeleteIcon fontSize="small" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Main Chat Area */}
        <div className="w-3/4 flex flex-col bg-gray-800 rounded-r-xl overflow-hidden">
          {/* Chat header */}
          <div className="px-6 py-4 border-b border-gray-700 flex justify-between items-center">
            <div className="flex items-center">
              <SmartToyIcon className="text-blue-400 mr-3" />
              <h2 className="text-lg font-semibold text-gray-200">{selectedConversation.title}</h2>
            </div>
          </div>
          
          {/* Messages area */}
          <div className="flex-1 overflow-y-auto px-4 py-2">
            {messages.map((message) => (
              <div 
                key={message.id} 
                className={`mb-6 ${message.role === 'user' ? 'pl-4' : 'pl-0'} ${message.isError ? 'border-l-2 border-red-500 pl-3' : ''} ${message.isStatusMessage ? 'border-l-2 border-blue-400 pl-3' : ''}`}
              >
                <div className="flex items-start mb-1">
                  {message.role === 'user' ? (
                    <div className="bg-blue-600 rounded-full p-2 mr-3">
                      <PersonIcon fontSize="small" className="text-white" />
                    </div>
                  ) : message.role === 'assistant' ? (
                    <div className="bg-green-600 rounded-full p-2 mr-3">
                      <SmartToyIcon fontSize="small" className="text-white" />
                    </div>
                  ) : (
                    <div className="bg-gray-600 rounded-full p-2 mr-3">
                      <InfoIcon fontSize="small" className="text-white" />
                    </div>
                  )}
                  <div 
                    className={`font-medium ${message.role === 'user' ? 'text-blue-400' : message.role === 'assistant' ? 'text-green-400' : 'text-gray-400'}`}
                  >
                    {message.role === 'user' ? 'You' : message.role === 'assistant' ? 'AI Assistant' : 'System'}
                  </div>
                </div>
                
                <div className={`prose prose-invert max-w-none pl-10 ${message.isError ? 'text-red-400' : ''}`}>
                  {renderMessageContent(message.content)}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          
          {/* Input area */}
          <div className="border-t border-gray-700 p-4">
            <div className="flex relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleInputKeyPress}
                placeholder="Ask a question or use /research or /quiz commands..."
                rows={1}
                className="flex-1 bg-gray-700 text-white px-4 py-3 pr-12 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
              <button
                onClick={handleSendMessage}
                disabled={!input.trim() || loading}
                className="absolute right-2 bottom-2 text-blue-400 hover:text-blue-300 disabled:text-gray-500 p-1 rounded-full"
              >
                {loading ? (
                  <div className="w-6 h-6 rounded-full border-2 border-t-blue-500 border-r-blue-500 border-b-blue-500 border-l-transparent animate-spin" />
                ) : (
                  <SendIcon />
                )}
              </button>
            </div>
            
            {/* Command suggestions */}
            <div className="flex gap-2 mt-2 text-xs text-gray-400">
              <span className="px-2 py-1 bg-gray-700 rounded-md hover:bg-gray-600 cursor-pointer">/research</span>
              <span className="px-2 py-1 bg-gray-700 rounded-md hover:bg-gray-600 cursor-pointer">/quiz</span>
              <span className="px-2 py-1 bg-gray-700 rounded-md hover:bg-gray-600 cursor-pointer">/help</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatPageTailwind; 