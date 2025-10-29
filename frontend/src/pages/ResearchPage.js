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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Fade,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SearchIcon from '@mui/icons-material/Search';
import ArticleIcon from '@mui/icons-material/Article';
import ReactMarkdown from 'react-markdown';
import ProcessTracker from '../components/ProcessTracker';
import {
  startResearch,
  getCurrentStatus,
  subscribeToStatusUpdates,
  subscribeToResearchComplete,
  subscribeToResearchError,
  initializeSocket,
  disconnectSocket,
} from '../services/apiService';

function ResearchPage() {
  const [topic, setTopic] = useState('');
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
    
    // Subscribe to research complete updates
    const unsubscribeComplete = subscribeToResearchComplete((data) => {
      setResults(data);
      setIsLoading(false);
    });
    
    // Subscribe to research error updates
    const unsubscribeError = subscribeToResearchError((data) => {
      setError(data.error);
      setIsLoading(false);
    });
    
    // Get current status on load
    getCurrentStatus()
      .then((data) => {
        setStatus(data);
        if (data.status === 'processing') {
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
    if (!topic.trim()) {
      setError('Please enter a research topic');
      return;
    }
    
    try {
      setIsLoading(true);
      setError('');
      setResults(null);
      await startResearch(topic);
    } catch (err) {
      setError(`Error starting research: ${err.message}`);
      setIsLoading(false);
    }
  };
  
  return (
    <Box sx={{ width: '100%' }}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={isLoading || results ? 5 : 12}>
          <Fade in={true}>
            <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Start New Research
              </Typography>
              
              <form onSubmit={handleSubmit}>
                <TextField
                  label="Research Topic"
                  variant="outlined"
                  fullWidth
                  multiline
                  rows={4}
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="Describe your research topic in detail. Include specific questions or areas you want to explore."
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
                  startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
                  disabled={isLoading || !topic.trim()}
                  sx={{ mb: 2 }}
                >
                  {isLoading ? 'Processing...' : 'Start Research'}
                </Button>
                
                <Typography variant="body2" color="text.secondary">
                  This will search through JIRA, Confluence, and Perforce data to gather relevant information.
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
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                      <Typography variant="h5" sx={{ fontWeight: 600 }}>
                        Research Results
                      </Typography>
                      <Button
                        variant="outlined"
                        startIcon={<ArticleIcon />}
                        href={`/api/download/${results.result_file}`}
                        target="_blank"
                      >
                        Download Full Report
                      </Button>
                    </Box>
                    
                    <Divider sx={{ mb: 3 }} />
                    
                    <Typography variant="h6" gutterBottom>
                      Summary
                    </Typography>
                    <Card variant="outlined" sx={{ mb: 3, bgcolor: 'background.paper' }}>
                      <CardContent>
                        <ReactMarkdown>
                          {results.summary || "No summary available."}
                        </ReactMarkdown>
                      </CardContent>
                    </Card>
                    
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle1" fontWeight={500}>
                          View Log Messages
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Box
                          component="pre"
                          sx={{
                            bgcolor: 'background.paper',
                            p: 2,
                            borderRadius: 1,
                            overflowX: 'auto',
                            fontFamily: 'monospace',
                            fontSize: '0.8rem',
                          }}
                        >
                          {status.log_messages.join('\n')}
                        </Box>
                      </AccordionDetails>
                    </Accordion>
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

export default ResearchPage; 