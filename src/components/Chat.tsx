import { useState, useEffect, useRef, FormEvent } from 'react'
import { 
  PaperAirplaneIcon, 
  Bars3Icon,
  ChartBarIcon,
  ClipboardDocumentListIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline'
import ReactMarkdown from 'react-markdown'
import { ConversationType, MessageType, StatusType } from '@/types'
import { classifyCommand } from '@/lib/utils'
import { 
  initializeSocket, 
  disconnectSocket,
  subscribeToStatusUpdates,
  subscribeToResearchComplete,
  subscribeToResearchError,
  subscribeToQuizComplete,
  subscribeToQuizError,
  startResearch,
  answerQuiz
} from '@/services/api'
import { ProgressTracker } from './ProgressTracker'
import { DetailedStatus } from './DetailedStatus'
import { ProcessVisualization } from './ProcessVisualization'

export function Chat({
  conversation,
  addMessage,
  updateConversation,
  toggleSidebar,
}: ChatProps) {
  const [input, setInput] = useState<string>('')
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [statusData, setStatusData] = useState<StatusType | null>(null)
  const [showDetailedStatus, setShowDetailedStatus] = useState<boolean>(false)
  const [showProcessVisualization, setShowProcessVisualization] = useState<boolean>(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // ... existing code for useEffect for scrolling ...

  // Initialize socket and subscribe to events
  useEffect(() => {
    const socket = initializeSocket()
    
    // Subscribe to status updates
    const unsubscribeStatus = subscribeToStatusUpdates((data) => {
      setStatusData(data)
      
      // If we have a status message, update it
      const statusMsgIndex = conversation.messages.findIndex(m => m.isStatusMessage)
      if (statusMsgIndex >= 0 && (data.status === 'processing' || data.status === 'starting')) {
        const statusMessage = formatStatusMessage(data)
        
        // Only update if the content has changed
        if (conversation.messages[statusMsgIndex].content !== statusMessage) {
          const updatedMessages = [...conversation.messages]
          updatedMessages[statusMsgIndex] = {
            ...updatedMessages[statusMsgIndex],
            content: statusMessage
          }
          updateConversation(conversation.id, { messages: updatedMessages })
        }
      }
      
      // If process completed or errored, end loading state
      if (data.status === 'completed' || data.status === 'error') {
        setIsLoading(false)
      }
    })
    
    // ... rest of the existing subscription code ...

    // Clean up subscriptions
    return () => {
      unsubscribeStatus()
      unsubscribeResearchComplete()
      unsubscribeResearchError()
      unsubscribeQuizComplete()
      unsubscribeQuizError()
    }
  }, [conversation.id, conversation.messages, addMessage, updateConversation])

  // ... existing functions ...

  // Render message content with markdown
  const renderMessageContent = (content: string, message: MessageType) => {
    // For status messages, we add the visualization components
    if (message.isStatusMessage && statusData) {
      return (
        <div>
          <ReactMarkdown className="prose prose-invert max-w-none mb-4">
            {content}
          </ReactMarkdown>
          
          <ProgressTracker status={statusData} isVisible={true} />
          
          <div className="flex flex-col space-y-2 mt-4">
            <button
              onClick={() => setShowDetailedStatus(!showDetailedStatus)}
              className="flex items-center justify-between px-3 py-2 bg-gray-800 hover:bg-gray-750 rounded-md text-sm text-gray-300 border border-gray-700"
            >
              <div className="flex items-center space-x-2">
                <ClipboardDocumentListIcon className="h-4 w-4 text-emerald-400" />
                <span>Detailed Status</span>
              </div>
              {showDetailedStatus ? 
                <ChevronUpIcon className="h-4 w-4" /> : 
                <ChevronDownIcon className="h-4 w-4" />
              }
            </button>
            
            <button
              onClick={() => setShowProcessVisualization(!showProcessVisualization)}
              className="flex items-center justify-between px-3 py-2 bg-gray-800 hover:bg-gray-750 rounded-md text-sm text-gray-300 border border-gray-700"
            >
              <div className="flex items-center space-x-2">
                <ChartBarIcon className="h-4 w-4 text-emerald-400" />
                <span>Process Visualization</span>
              </div>
              {showProcessVisualization ? 
                <ChevronUpIcon className="h-4 w-4" /> : 
                <ChevronDownIcon className="h-4 w-4" />
              }
            </button>
          </div>
          
          <DetailedStatus status={statusData} isVisible={showDetailedStatus} />
          <ProcessVisualization status={statusData} isVisible={showProcessVisualization} />
          
          {/* Add a refresh button to manually update status */}
          <div className="flex justify-center mt-4">
            <button 
              className="flex items-center space-x-1 text-xs text-emerald-400 hover:text-emerald-300 px-2 py-1 rounded-md bg-gray-800 hover:bg-gray-750"
              onClick={() => subscribeToStatusUpdates((data) => setStatusData(data))}
            >
              <ArrowPathIcon className="h-3 w-3" />
              <span>Refresh Status</span>
            </button>
          </div>
        </div>
      )
    }
    
    // Otherwise, just render the markdown
    return (
      <ReactMarkdown className="prose prose-invert max-w-none">
        {content}
      </ReactMarkdown>
    )
  }

  // ... rest of the function including the welcome message and return ...

  return (
    <div className="flex-1 flex flex-col">
      {/* Mobile header */}
      <div className="md:hidden flex items-center justify-between p-4 border-b border-gray-800">
        <button 
          onClick={toggleSidebar}
          className="text-gray-400 hover:text-white"
        >
          <Bars3Icon className="w-6 h-6" />
        </button>
        <h1 className="text-lg font-semibold text-white">{conversation.title}</h1>
        <div className="w-6"></div> {/* Empty div for alignment */}
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {conversation.messages.map((message) => (
          <div 
            key={message.id}
            className={`mb-6 ${message.isError ? 'border-l-2 border-red-500 pl-3' : ''} ${message.isStatusMessage ? 'border-l-2 border-emerald-500 pl-3' : ''}`}
          >
            <div className="flex items-center mb-2">
              <div 
                className={`w-8 h-8 rounded-full flex items-center justify-center mr-2 ${
                  message.role === 'user' 
                    ? 'bg-blue-600' 
                    : message.role === 'assistant' 
                    ? 'bg-emerald-600' 
                    : 'bg-gray-600'
                }`}
              >
                {message.role === 'user' ? (
                  <span className="text-white font-semibold">U</span>
                ) : message.role === 'assistant' ? (
                  <span className="text-white font-semibold">AI</span>
                ) : (
                  <span className="text-white font-semibold">S</span>
                )}
              </div>
              <div className={`font-medium ${
                message.role === 'user' 
                  ? 'text-blue-400' 
                  : message.role === 'assistant' 
                  ? 'text-emerald-400' 
                  : 'text-gray-400'
              }`}>
                {message.role === 'user' ? 'You' : message.role === 'assistant' ? 'AI Assistant' : 'System'}
              </div>
            </div>
            
            <div className={`ml-10 ${message.isError ? 'text-red-400' : message.isStatusMessage ? 'text-gray-300' : 'text-gray-200'}`}>
              {renderMessageContent(message.content, message)}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      {/* ... rest of the return statement ... */}
    </div>
  )
} 