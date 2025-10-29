import React from 'react';
import { Box, Typography, Paper, Grid, Button, Card, CardContent, CardActions, Divider } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import SearchIcon from '@mui/icons-material/Search';
import QuizIcon from '@mui/icons-material/Quiz';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import SecurityIcon from '@mui/icons-material/Security';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ChatIcon from '@mui/icons-material/Chat';

function HomePage() {
  const navigate = useNavigate();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Hero Section */}
      <Paper
        elevation={3}
        sx={{
          position: 'relative',
          borderRadius: 4,
          mb: 5,
          py: 6,
          px: 5,
          background: 'linear-gradient(120deg, #1a237e 0%, #3949ab 35%, #42a5f5 100%)',
          color: 'white',
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          alignItems: 'center',
          justifyContent: 'space-between',
          overflow: 'hidden',
          boxShadow: '0 10px 30px rgba(25, 118, 210, 0.15)',
        }}
      >
        <Box sx={{ zIndex: 2, flex: 1 }}>
          <Typography variant="h3" gutterBottom fontWeight={800} sx={{
            textShadow: '0 2px 10px rgba(0,0,0,0.2)',
            mb: 2,
          }}>
            Ollama Deep Researcher
          </Typography>
          <Typography variant="h6" sx={{ mb: 3, opacity: 0.9, fontWeight: 500 }}>
            Advanced enterprise research using Ollama models
          </Typography>
          <Typography variant="body1" sx={{ mb: 4, maxWidth: '600px', lineHeight: 1.7 }}>
            Analyze documentation, extract insights, and answer complex questions using powerful local LLMs with RAG technology.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="large"
              onClick={() => navigate('/')}
              startIcon={<ChatIcon />}
              sx={{ 
                bgcolor: 'white', 
                color: '#1976d2',
                px: 4,
                py: 1.5,
                borderRadius: 2,
                fontWeight: 600,
                boxShadow: '0 6px 12px rgba(0,0,0,0.15)',
                textTransform: 'none',
                '&:hover': {
                  bgcolor: 'rgba(255, 255, 255, 0.9)',
                  transform: 'translateY(-2px)',
                  boxShadow: '0 8px 16px rgba(0,0,0,0.2)',
                },
                transition: 'all 0.2s',
              }}
            >
              Start Chatting
            </Button>
          </Box>
        </Box>
        
        <Box 
          sx={{ 
            display: { xs: 'none', md: 'flex' },
            justifyContent: 'center',
            alignItems: 'center',
            flex: 1,
            position: 'relative',
            height: '100%',
          }}
        >
          <AutoAwesomeIcon 
            sx={{ 
              fontSize: 280, 
              opacity: 0.15, 
              position: 'absolute',
              right: '-40px',
              top: '-60px',
              transform: 'rotate(15deg)',
            }} 
          />
        </Box>
      </Paper>

      {/* Features Section */}
      <Typography variant="h4" sx={{ mb: 4, fontWeight: 700, textAlign: 'center', color: '#1a237e' }}>
        Key Features
      </Typography>
      
      <Grid container spacing={4}>
        {[
          {
            title: 'Deep Enterprise Research',
            description: 'Search through and analyze JIRA, Confluence, and Perforce data to extract valuable insights.',
            icon: <SearchIcon fontSize="large" sx={{ color: '#3949ab' }} />,
            command: '/research',
            color: '#e3f2fd'
          },
          {
            title: 'Knowledge Quiz Answering',
            description: 'Answer quizzes and knowledge checks based on the research findings with evidence-based responses.',
            icon: <QuizIcon fontSize="large" sx={{ color: '#7b1fa2' }} />,
            command: '/quiz',
            color: '#f3e5f5'
          },
          {
            title: 'ChatGPT-Style Interface',
            description: 'Interact with a conversational AI interface to research enterprise data with chat history and context.',
            icon: <ChatIcon fontSize="large" sx={{ color: '#0288d1' }} />,
            command: '/help',
            color: '#e1f5fe'
          },
          {
            title: 'Comprehensive Analysis',
            description: 'Get detailed summaries, technical validations, and security findings from your enterprise data.',
            icon: <AnalyticsIcon fontSize="large" sx={{ color: '#00897b' }} />,
            command: '/research',
            color: '#e0f2f1'
          },
          {
            title: 'Privacy & Security',
            description: 'All processing is done locally using Ollama models, ensuring your sensitive data never leaves your system.',
            icon: <SecurityIcon fontSize="large" sx={{ color: '#c62828' }} />,
            command: '',
            color: '#ffebee'
          }
        ].map((feature, index) => (
          <Grid item xs={12} sm={6} md={index > 2 ? 6 : 4} key={index}>
            <Card 
              elevation={2}
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'transform 0.3s, box-shadow 0.3s',
                borderRadius: 3,
                overflow: 'hidden',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: '0 12px 24px rgba(0,0,0,0.12)',
                }
              }}
            >
              <CardContent sx={{ pt: 4, pb: 3, flexGrow: 1, bgcolor: feature.color }}>
                <Box 
                  sx={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    mb: 3,
                    p: 2,
                    borderRadius: '50%',
                    width: 70,
                    height: 70,
                    bgcolor: 'white',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                    mx: 'auto',
                  }}
                >
                  {feature.icon}
                </Box>
                <Typography gutterBottom variant="h5" component="div" align="center" fontWeight={700}>
                  {feature.title}
                </Typography>
                <Typography variant="body1" sx={{ mt: 2, textAlign: 'center', fontSize: '1rem', lineHeight: 1.6 }}>
                  {feature.description}
                </Typography>
                {feature.command && (
                  <Typography variant="subtitle2" color="primary" align="center" sx={{ mt: 2, display: 'block', fontFamily: 'monospace', letterSpacing: 0.5 }}>
                    Command: <code style={{ backgroundColor: 'rgba(25, 118, 210, 0.08)', padding: '4px 8px', borderRadius: '4px' }}>{feature.command}</code>
                  </Typography>
                )}
              </CardContent>
              <Divider />
              <CardActions sx={{ justifyContent: 'center', py: 1.5 }}>
                <Button 
                  size="medium" 
                  variant="contained"
                  sx={{
                    textTransform: 'none',
                    borderRadius: 8,
                    px: 3,
                    fontWeight: 600,
                    boxShadow: 'none',
                    '&:hover': {
                      boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                    }
                  }}
                  onClick={() => {
                    navigate('/');
                    // If you need to pre-populate the chat input, you'd need to use state management
                    // For now, just navigate to the chat page
                  }}
                >
                  Try in Chat
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

export default HomePage; 