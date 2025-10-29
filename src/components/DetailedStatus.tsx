'use client'

import { StatusType } from '@/types'
import { useState } from 'react'

interface DetailedStatusProps {
  status: StatusType | null
  isVisible: boolean
}

export function DetailedStatus({ status, isVisible }: DetailedStatusProps) {
  const [showAllLogs, setShowAllLogs] = useState(false)
  
  if (!isVisible || !status) return null
  
  // Status indicator color based on status.status
  const getStatusColor = (statusValue: string) => {
    switch(statusValue) {
      case 'idle': return 'bg-gray-500'
      case 'starting': return 'bg-blue-500'
      case 'processing': return 'bg-emerald-500 animate-pulse'
      case 'completed': return 'bg-green-500'
      case 'error': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }
  
  // Format the elapsed time
  const formatElapsedTime = () => {
    // Return a placeholder for now
    return "Calculating..."
  }
  
  // Get the estimated time remaining
  const getEstimatedTimeRemaining = () => {
    const progress = status.progress || 0
    if (progress >= 100) return 'Complete'
    if (progress <= 0) return 'Calculating...'
    
    // Simple calculation - if we're at 25% and 5 minutes have passed,
    // then we estimate 15 more minutes
    return 'Calculating...'
  }
  
  return (
    <div className="mt-2 bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      <div className="p-3 bg-gray-750 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200 flex items-center">
            <div className={`w-2 h-2 rounded-full mr-2 ${getStatusColor(status.status)}`}></div>
            Process Status: <span className="ml-1.5 text-emerald-400 capitalize">{status.status}</span>
          </h3>
          <div className="text-xs text-gray-400 flex items-center space-x-2">
            <span>Elapsed: {formatElapsedTime()}</span>
            <span>|</span>
            <span>ETA: {getEstimatedTimeRemaining()}</span>
          </div>
        </div>
      </div>
      
      <div className="p-3">
        <div className="mb-3">
          <div className="flex justify-between mb-1">
            <span className="text-xs text-gray-400">Progress</span>
            <span className="text-xs font-medium text-emerald-400">{status.progress}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1.5">
            <div 
              className="bg-emerald-600 h-1.5 rounded-full transition-all duration-500"
              style={{ width: `${status.progress}%` }}
            ></div>
          </div>
        </div>
        
        <div className="mb-3">
          <div className="text-xs text-gray-400 mb-1">Current Operation</div>
          <div className="bg-gray-700 p-2 rounded text-xs text-white font-mono">
            {status.current_step || 'Waiting...'}
          </div>
        </div>
        
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-gray-400">Process Logs</span>
            <button 
              onClick={() => setShowAllLogs(!showAllLogs)}
              className="text-[10px] text-emerald-400 hover:text-emerald-300"
            >
              {showAllLogs ? 'Show Recent' : 'Show All'}
            </button>
          </div>
          <div className="bg-gray-900 p-2 rounded text-xs text-gray-300 font-mono max-h-40 overflow-y-auto">
            {status.log_messages && status.log_messages.length > 0 ? (
              <ul className="space-y-0.5">
                {(showAllLogs ? status.log_messages : status.log_messages.slice(-5)).map((log, index) => (
                  <li key={index} className="py-0.5 border-b border-gray-800 last:border-0">
                    {log}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500 italic">No logs available</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
} 