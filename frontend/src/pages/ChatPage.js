import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Divider,
  IconButton,
  Grid,
  Avatar,
  Chip,
  CircularProgress,
  InputAdornment,
  alpha,
  useTheme,
  Tooltip,
} from '@mui/material';
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

function ChatPage() {
  const theme = useTheme();
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
          content: 'Welcome to Ollama Deep Researcher Chat. I can help you with research and quiz answering. Try these commands:\n\n' +
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
        content: 'Welcome to Ollama Deep Researcher Chat. I can help you with research and quiz answering. Try these commands:\n\n' +
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
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: 'calc(100vh - 120px)', 
        maxWidth: '1600px', 
        mx: 'auto',
        px: 2,
        py: 3 
      }}>
        <Grid container spacing={0} sx={{ height: '100%' }}>
          {/* Sidebar */}
          <Grid item xs={3} sx={{ 
            borderRight: `1px solid ${theme.palette.divider}`,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            borderRadius: '16px 0 0 16px',
            boxShadow: '0 3px 10px rgba(0,0,0,0.08)',
            overflow: 'hidden',
          }}>
            <Button 
              variant="contained" 
              startIcon={<AddIcon />}
              fullWidth
              onClick={handleNewChat}
              sx={{ 
                m: 2, 
                mb: 2, 
                borderRadius: '12px',
                textTransform: 'none',
                fontWeight: 600,
                py: 1.2,
                boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                background: 'linear-gradient(90deg, #1976d2, #2196F3)',
                '&:hover': {
                  background: 'linear-gradient(90deg, #1565c0, #1976d2)',
                  boxShadow: '0 6px 12px rgba(0,0,0,0.15)',
                }
              }}
            >
              New Chat
            </Button>
            
            <Typography 
              variant="subtitle2" 
              color="text.secondary" 
              sx={{ 
                px: 2.5, 
                py: 1.5, 
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <HistoryIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 1.5, opacity: 0.7 }} />
              Previous Conversations
            </Typography>
            
            <List sx={{ overflowY: 'auto', flex: 1, px: 1.5, py: 1 }}>
              {conversations.map((conversation) => (
                <ListItemButton 
                  key={conversation.id}
                  onClick={() => setSelectedConversation(conversation)}
                  sx={{ 
                    borderRadius: '12px',
                    mb: 1,
                    py: 1.8,
                    pl: 2,
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      backgroundColor: alpha(theme.palette.primary.main, 0.08),
                      transform: 'translateY(-2px)',
                    }
                  }}
                >
                  <ListItemText 
                    primary={conversation.title} 
                    secondary={conversation.updated}
                    primaryTypographyProps={{
                      noWrap: true,
                      fontWeight: 600,
                      fontSize: '0.95rem',
                    }}
                    secondaryTypographyProps={{
                      fontSize: '0.75rem',
                      color: alpha(theme.palette.text.secondary, 0.7)
                    }}
                  />
                  <Tooltip title="Delete conversation">
                    <IconButton 
                      edge="end" 
                      size="small"
                      onClick={(e) => handleDeleteConversation(e, conversation.id)}
                      sx={{ 
                        opacity: 0.5, 
                        '&:hover': { 
                          opacity: 1,
                          color: theme.palette.error.main,
                          backgroundColor: alpha(theme.palette.error.main, 0.1),
                        } 
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </ListItemButton>
              ))}
            </List>
          </Grid>
          
          {/* Empty State */}
          <Grid item xs={9} sx={{ 
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100%',
            bgcolor: alpha(theme.palette.background.paper, 0.4),
            borderRadius: '0 16px 16px 0',
            boxShadow: '0 3px 10px rgba(0,0,0,0.06)',
            p: 4,
          }}>
            <Box
              component="img"
              src="/assets/logo192.png"
              alt="Ollama Logo"
              sx={{
                width: 100,
                height: 100,
                mb: 4,
                opacity: 0.7,
                filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))',
              }}
            />
            <Typography variant="h4" align="center" gutterBottom fontWeight={700} color="primary.main">
              Ollama Deep Researcher
            </Typography>
            <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4, maxWidth: 600 }}>
              I can help you research enterprise information from JIRA, Confluence, and code repositories. Select a conversation from the sidebar or start a new chat.
            </Typography>
            
            <Grid container spacing={3} justifyContent="center" sx={{ maxWidth: 900, mt: 2 }}>
              <Grid item xs={12} sm={4}>
                <Paper
                  elevation={2}
                  sx={{
                    p: 3,
                    borderRadius: 3,
                    height: '100%',
                    transition: 'all 0.2s',
                    textAlign: 'center',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                    },
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Box 
                    sx={{
                      bgcolor: alpha(theme.palette.primary.main, 0.1), 
                      borderRadius: '50%',
                      p: 2,
                      mb: 2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <SearchIcon color="primary" fontSize="large" />
                  </Box>
                  <Typography variant="h6" gutterBottom fontWeight={600}>
                    Deep Research
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Search enterprise repositories
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Paper
                  elevation={2}
                  sx={{
                    p: 3,
                    borderRadius: 3,
                    height: '100%',
                    transition: 'all 0.2s',
                    textAlign: 'center',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                    },
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Box 
                    sx={{
                      bgcolor: alpha(theme.palette.secondary.main, 0.1), 
                      borderRadius: '50%',
                      p: 2,
                      mb: 2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <QuizIcon color="secondary" fontSize="large" />
                  </Box>
                  <Typography variant="h6" gutterBottom fontWeight={600}>
                    Quiz Answering
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Get evidence-based answers
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Paper
                  elevation={2}
                  sx={{
                    p: 3,
                    borderRadius: 3,
                    height: '100%',
                    transition: 'all 0.2s',
                    textAlign: 'center',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                    },
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Box 
                    sx={{
                      bgcolor: alpha(theme.palette.info.main, 0.1), 
                      borderRadius: '50%',
                      p: 2,
                      mb: 2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <HelpIcon color="info" fontSize="large" />
                  </Box>
                  <Typography variant="h6" gutterBottom fontWeight={600}>
                    Chat Assistant
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Interactive research help
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
            
            <Button 
              variant="contained" 
              size="large"
              startIcon={<AddIcon />}
              onClick={handleNewChat}
              sx={{ 
                mt: 6, 
                borderRadius: '12px',
                textTransform: 'none',
                fontWeight: 600,
                px: 4,
                py: 1.5,
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                background: 'linear-gradient(90deg, #1976d2, #2196F3)',
                '&:hover': {
                  background: 'linear-gradient(90deg, #1565c0, #1976d2)',
                  boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
                  transform: 'translateY(-2px)',
                }
              }}
            >
              Start New Research
            </Button>
          </Grid>
        </Grid>
      </Box>
    );
  }

  // Return the chat interface if a conversation is selected
  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: 'calc(100vh - 120px)', 
      maxWidth: '1600px', 
      mx: 'auto',
      px: 2,
      py: 3 
    }}>
      <Grid container spacing={0} sx={{ height: '100%' }}>
        {/* Sidebar */}
        <Grid item xs={3} sx={{ 
          borderRight: `1px solid ${theme.palette.divider}`,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          bgcolor: alpha(theme.palette.background.paper, 0.7),
          borderRadius: '16px 0 0 16px',
          boxShadow: '0 3px 10px rgba(0,0,0,0.08)',
          overflow: 'hidden',
        }}>
          <Button 
            variant="contained" 
            startIcon={<AddIcon />}
            fullWidth
            onClick={handleNewChat}
            sx={{ 
              m: 2, 
              mb: 2, 
              borderRadius: '12px',
              textTransform: 'none',
              fontWeight: 600,
              py: 1.2,
              boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
              background: 'linear-gradient(90deg, #1976d2, #2196F3)',
              '&:hover': {
                background: 'linear-gradient(90deg, #1565c0, #1976d2)',
                boxShadow: '0 6px 12px rgba(0,0,0,0.15)',
              }
            }}
          >
            New Chat
          </Button>
          
          <Typography 
            variant="subtitle2" 
            color="text.secondary" 
            sx={{ 
              px: 2.5, 
              py: 1.5, 
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <HistoryIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 1.5, opacity: 0.7 }} />
            Previous Conversations
          </Typography>
          
          <List sx={{ overflowY: 'auto', flex: 1, px: 1.5, py: 1 }}>
            {conversations.map((conversation) => (
              <ListItemButton 
                key={conversation.id}
                selected={selectedConversation?.id === conversation.id}
                onClick={() => setSelectedConversation(conversation)}
                sx={{ 
                  borderRadius: '12px',
                  mb: 1,
                  py: 1.8,
                  pl: 2,
                  transition: 'all 0.2s ease',
                  '&.Mui-selected': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.12),
                    '&:hover': {
                      backgroundColor: alpha(theme.palette.primary.main, 0.18),
                    }
                  },
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.08),
                    transform: 'translateY(-2px)',
                  }
                }}
              >
                <ListItemText 
                  primary={conversation.title} 
                  secondary={conversation.updated}
                  primaryTypographyProps={{
                    noWrap: true,
                    fontWeight: selectedConversation?.id === conversation.id ? 700 : 600,
                    fontSize: '0.95rem',
                    color: selectedConversation?.id === conversation.id ? theme.palette.primary.main : 'inherit',
                  }}
                  secondaryTypographyProps={{
                    fontSize: '0.75rem',
                    color: alpha(theme.palette.text.secondary, 0.7)
                  }}
                />
                <Tooltip title="Delete conversation">
                  <IconButton 
                    edge="end" 
                    size="small"
                    onClick={(e) => handleDeleteConversation(e, conversation.id)}
                    sx={{ 
                      opacity: 0.5, 
                      '&:hover': { 
                        opacity: 1,
                        color: theme.palette.error.main,
                        backgroundColor: alpha(theme.palette.error.main, 0.1),
                      } 
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </ListItemButton>
            ))}
          </List>
        </Grid>
        
        {/* Chat Window */}
        <Grid item xs={9} sx={{ 
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          bgcolor: alpha(theme.palette.background.paper, 0.4),
          borderRadius: '0 16px 16px 0',
          boxShadow: '0 3px 10px rgba(0,0,0,0.06)',
        }}>
          {/* Chat Header */}
          <Box sx={{ 
            p: 2.5, 
            borderBottom: `1px solid ${alpha(theme.palette.divider, 0.6)}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <Typography variant="h6" fontWeight={600}>
              {selectedConversation?.title}
            </Typography>
            <Chip 
              icon={<UpdateIcon fontSize="small" />} 
              label={selectedConversation?.updated} 
              size="small"
              sx={{ 
                borderRadius: '8px',
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                color: theme.palette.primary.main,
                fontWeight: 500,
              }}
            />
          </Box>
          
          {/* Messages Area */}
          <Box sx={{ 
            flex: 1, 
            p: 3, 
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 2.5,
            bgcolor: alpha(theme.palette.background.default, 0.5),
          }}>
            {messages.map((message) => (
              <Box 
                key={message.id} 
                sx={{ 
                  display: 'flex',
                  flexDirection: 'column',
                  alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: message.role === 'system' ? '100%' : '85%',
                  width: message.role === 'system' ? '100%' : 'auto',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
                  {message.role !== 'user' && (
                    <Avatar 
                      sx={{ 
                        bgcolor: message.role === 'assistant' 
                          ? alpha(theme.palette.primary.main, 0.8) 
                          : message.isError 
                            ? theme.palette.error.main 
                            : alpha(theme.palette.grey[500], 0.8),
                        width: 38,
                        height: 38,
                      }}
                    >
                      {message.role === 'assistant' ? (
                        <SmartToyIcon fontSize="small" />
                      ) : message.isError ? (
                        <ErrorIcon fontSize="small" />
                      ) : (
                        <InfoIcon fontSize="small" />
                      )}
                    </Avatar>
                  )}
                  
                  <Paper 
                    elevation={message.role === 'system' ? 0 : 1}
                    sx={{ 
                      p: message.role === 'system' ? 2 : 2.5,
                      borderRadius: message.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                      bgcolor: message.role === 'user' 
                        ? 'primary.main' 
                        : message.role === 'assistant' 
                          ? alpha(theme.palette.primary.light, 0.08)
                          : message.isError 
                            ? alpha(theme.palette.error.light, 0.1)
                            : message.isStatusMessage
                              ? alpha(theme.palette.info.light, 0.1)
                              : alpha(theme.palette.grey[100], 0.7),
                      color: message.role === 'user' ? 'white' : 'text.primary',
                      maxWidth: '100%',
                      overflow: 'hidden',
                      boxShadow: message.role === 'user' 
                        ? '0 4px 12px rgba(25, 118, 210, 0.2)' 
                        : message.role === 'assistant'
                          ? '0 2px 8px rgba(0, 0, 0, 0.08)'
                          : 'none',
                      '& a': {
                        color: message.role === 'user' ? 'inherit' : theme.palette.primary.main,
                        textDecoration: 'none',
                        fontWeight: 600,
                        '&:hover': {
                          textDecoration: 'underline',
                        }
                      },
                      '& code': {
                        backgroundColor: message.role === 'user' 
                          ? alpha(theme.palette.common.white, 0.15) 
                          : alpha(theme.palette.grey[200], 0.5),
                        padding: '2px 5px',
                        borderRadius: '4px',
                        fontSize: '0.85em',
                      },
                      '& pre': {
                        backgroundColor: alpha(theme.palette.grey[900], 0.05),
                        padding: 2,
                        borderRadius: '8px',
                        overflow: 'auto',
                        fontSize: '0.9em',
                        '& code': {
                          backgroundColor: 'transparent',
                        }
                      },
                    }}
                  >
                    {renderMessageContent(message.content)}
                  </Paper>
                  
                  {message.role === 'user' && (
                    <Avatar 
                      sx={{ 
                        bgcolor: theme.palette.primary.dark,
                        width: 38,
                        height: 38,
                      }}
                    >
                      <PersonIcon fontSize="small" />
                    </Avatar>
                  )}
                </Box>
              </Box>
            ))}
            
            {loading && (
              <Box 
                sx={{ 
                  display: 'flex', 
                  justifyContent: 'center',
                  p: 2,
                  color: 'text.secondary',
                }}
              >
                <CircularProgress size={32} color="primary" thickness={4} />
              </Box>
            )}
            
            <div ref={messagesEndRef} />
          </Box>
          
          {/* Input Area */}
          <Box sx={{ 
            p: 2.5, 
            borderTop: `1px solid ${alpha(theme.palette.divider, 0.6)}`,
          }}>
            <TextField
              fullWidth
              placeholder="Type a message or use commands like /research, /quiz, /help..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleInputKeyPress}
              multiline
              maxRows={4}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      color="primary"
                      onClick={handleSendMessage}
                      disabled={!input.trim() || loading}
                      sx={{ 
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        '&:hover': {
                          bgcolor: alpha(theme.palette.primary.main, 0.2),
                        },
                        transition: 'all 0.2s',
                        mr: -1,
                      }}
                    >
                      <SendIcon />
                    </IconButton>
                  </InputAdornment>
                ),
                sx: {
                  p: 1.5,
                  borderRadius: '12px',
                  backgroundColor: theme.palette.background.paper,
                  boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
                  '&.Mui-focused': {
                    boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
                  },
                  transition: 'all 0.2s',
                }
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: 'transparent',
                  },
                  '&:hover fieldset': {
                    borderColor: alpha(theme.palette.primary.main, 0.3),
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: alpha(theme.palette.primary.main, 0.6),
                  },
                },
              }}
            />
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ChatPage; 