import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Grid,
  Alert,
  Card,
  CardContent,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  Fade,
  Chip,
} from '@mui/material';
import QuizIcon from '@mui/icons-material/Quiz';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ProcessTracker from '../components/ProcessTracker';
import {
  answerQuiz,
  getCurrentStatus,
  subscribeToStatusUpdates,
  subscribeToQuizComplete,
  subscribeToQuizError,
  initializeSocket,
  disconnectSocket,
} from '../services/apiService';

function QuizPage() {
  const [quizText, setQuizText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState({
    status: 'idle',
    current_step: '',
    progress: 0,
    log_messages: [],
  });

  useEffect(() => {
    // Initialize WebSocket connection
    initializeSocket();
    
    // Subscribe to status updates
    const unsubscribeStatus = subscribeToStatusUpdates((data) => {
      setStatus(data);
      if (data.status === 'processing' && isLoading === false) {
        setIsLoading(true);
      } else if (data.status === 'completed' || data.status === 'error') {
        setIsLoading(false);
      }
    });
    
    // Subscribe to quiz complete updates
    const unsubscribeComplete = subscribeToQuizComplete((data) => {
      setResults(data);
      setIsLoading(false);
    });
    
    // Subscribe to quiz error updates
    const unsubscribeError = subscribeToQuizError((data) => {
      setError(data.error);
      setIsLoading(false);
    });
    
    // Get current status on load
    getCurrentStatus()
      .then((data) => {
        setStatus(data);
        if (data.status === 'processing' && data.current_step.includes('quiz')) {
          setIsLoading(true);
        }
      })
      .catch((err) => {
        console.error('Error getting status:', err);
      });
    
    // Clean up on unmount
    return () => {
      unsubscribeStatus();
      unsubscribeComplete();
      unsubscribeError();
      disconnectSocket();
    };
  }, []);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!quizText.trim()) {
      setError('Please enter quiz questions');
      return;
    }
    
    try {
      setIsLoading(true);
      setError('');
      setResults(null);
      await answerQuiz(quizText);
    } catch (err) {
      setError(`Error answering quiz: ${err.message}`);
      setIsLoading(false);
    }
  };
  
  const renderQuizAnswers = () => {
    if (!results || !results.answers || results.answers.length === 0) {
      return (
        <Alert severity="info">
          No answers available.
        </Alert>
      );
    }
    
    return (
      <List>
        {results.answers.map((answer, index) => (
          <React.Fragment key={index}>
            <ListItem alignItems="flex-start" sx={{ flexDirection: 'column', py: 2 }}>
              <Box sx={{ width: '100%', mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                  Question {index + 1}:
                </Typography>
                <Typography variant="body1" color="text.primary" sx={{ mb: 1 }}>
                  {answer.question}
                </Typography>
                
                {answer.options && answer.options.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Options:
                    </Typography>
                    <List dense disablePadding>
                      {answer.options.map((option, optIndex) => (
                        <ListItem 
                          key={optIndex} 
                          dense 
                          disablePadding 
                          sx={{ 
                            mb: 0.5,
                            color: answer.answer === option ? 'success.main' : 'text.primary',
                            fontWeight: answer.answer === option ? 700 : 400,
                          }}
                        >
                          <ListItemText>
                            {answer.answer === option && (
                              <CheckCircleIcon 
                                fontSize="small" 
                                color="success"
                                sx={{ mr: 1, verticalAlign: 'middle' }}
                              />
                            )}
                            {option}
                          </ListItemText>
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </Box>
              
              <Box sx={{ width: '100%' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Answer:
                </Typography>
                <Card variant="outlined" sx={{ mb: 1, bgcolor: 'background.paper' }}>
                  <CardContent sx={{ py: 1, px: 2, '&:last-child': { pb: 1 } }}>
                    <Typography variant="body1" fontWeight={500} color="success.main">
                      {answer.answer || "No answer provided"}
                    </Typography>
                  </CardContent>
                </Card>
                
                <Typography variant="subtitle2" gutterBottom>
                  Justification:
                </Typography>
                <Card variant="outlined" sx={{ bgcolor: 'background.paper' }}>
                  <CardContent>
                    <Typography variant="body2">
                      {answer.justification || "No justification provided"}
                    </Typography>
                    
                    {answer.sources && answer.sources.length > 0 && (
                      <Box mt={2}>
                        <Typography variant="caption" fontWeight={600}>
                          Sources:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 0.5 }}>
                          {answer.sources.map((source, srcIndex) => (
                            <Chip 
                              key={srcIndex} 
                              size="small" 
                              label={source} 
                              color="info" 
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Box>
            </ListItem>
            {index < results.answers.length - 1 && <Divider />}
          </React.Fragment>
        ))}
      </List>
    );
  };
  
  return (
    <Box sx={{ width: '100%' }}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={isLoading || results ? 5 : 12}>
          <Fade in={true}>
            <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Answer Quiz Questions
              </Typography>
              
              <form onSubmit={handleSubmit}>
                <TextField
                  label="Quiz Questions"
                  variant="outlined"
                  fullWidth
                  multiline
                  rows={8}
                  value={quizText}
                  onChange={(e) => setQuizText(e.target.value)}
                  placeholder="Paste your quiz questions here. Include all questions and multiple-choice options if available."
                  sx={{ mb: 3 }}
                  disabled={isLoading}
                />
                
                {error && (
                  <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                  </Alert>
                )}
                
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <QuizIcon />}
                  disabled={isLoading || !quizText.trim()}
                  sx={{ mb: 2 }}
                >
                  {isLoading ? 'Processing...' : 'Answer Quiz'}
                </Button>
                
                <Typography variant="body2" color="text.secondary">
                  Make sure test_data.json is available with the necessary data for quiz answering.
                </Typography>
              </form>
            </Paper>
          </Fade>
        </Grid>
        
        {(isLoading || results) && (
          <Grid item xs={12} md={7}>
            <Fade in={true}>
              <Box>
                <ProcessTracker
                  status={status.status}
                  currentStep={status.current_step}
                  progress={status.progress}
                  logMessages={status.log_messages}
                />
                
                {results && (
                  <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
                    <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                      Quiz Answers
                    </Typography>
                    
                    <Divider sx={{ mb: 3 }} />
                    
                    {renderQuizAnswers()}
                  </Paper>
                )}
              </Box>
            </Fade>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default QuizPage; 