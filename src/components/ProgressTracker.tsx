'use client'

import { useState, useEffect } from 'react'
import { StatusType } from '@/types'

interface ProgressTrackerProps {
  status: StatusType | null
  isVisible: boolean
}

export function ProgressTracker({ status, isVisible }: ProgressTrackerProps) {
  if (!isVisible || !status) return null
  
  // Define the research process steps
  const researchSteps = [
    { id: 'initializing', label: 'Initializing', icon: '🔍' },
    { id: 'search_query_generation', label: 'Query Generation', icon: '✍️' },
    { id: 'jira_search', label: 'JIRA Research', icon: '📊' },
    { id: 'confluence_search', label: 'Confluence Research', icon: '📄' },
    { id: 'perforce_search', label: 'Perforce Research', icon: '🧩' },
    { id: 'process_results', label: 'Processing Results', icon: '⚙️' },
    { id: 'generate_summary', label: 'Generating Summary', icon: '📝' },
    { id: 'finalize', label: 'Finalizing', icon: '✅' }
  ]
  
  // Determine current step index based on status.current_step
  const currentStepIndex = researchSteps.findIndex(
    step => status.current_step?.toLowerCase().includes(step.id)
  )
  
  // If current step not found in our predefined steps, default to progress percentage
  const normalizedStepIndex = currentStepIndex === -1 
    ? Math.floor(status.progress / (100 / researchSteps.length))
    : currentStepIndex
  
  return (
    <div className="mt-4 mb-4 p-4 bg-gray-800 rounded-lg border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-md font-medium text-white">Research Progress</h3>
        <span className="text-sm font-medium text-emerald-400">
          {status.progress}%
        </span>
      </div>
      
      {/* Progress bar */}
      <div className="w-full bg-gray-700 rounded-full h-2.5 mb-4">
        <div 
          className="bg-emerald-600 h-2.5 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${status.progress}%` }}
        ></div>
      </div>
      
      {/* Steps indicator */}
      <div className="grid grid-cols-4 md:grid-cols-8 gap-2 mb-4">
        {researchSteps.map((step, index) => (
          <div 
            key={step.id}
            className={`flex flex-col items-center justify-center p-2 rounded-md ${
              index <= normalizedStepIndex 
                ? index === normalizedStepIndex 
                  ? 'bg-emerald-900/50 border border-emerald-500/50 animate-pulse' 
                  : 'bg-emerald-800/20 border border-emerald-500/20' 
                : 'bg-gray-700/50 border border-gray-600/50'
            }`}
          >
            <div className="text-lg mb-1">{step.icon}</div>
            <div className="text-[9px] md:text-xs text-center text-gray-300 whitespace-nowrap overflow-hidden text-ellipsis w-full">
              {step.label}
            </div>
          </div>
        ))}
      </div>
      
      {/* Current step details */}
      <div className="mb-3">
        <h4 className="text-sm font-medium text-emerald-300 mb-1">Current Step</h4>
        <div className="text-sm text-white bg-gray-700 rounded-md p-2">
          {status.current_step || 'Processing...'}
        </div>
      </div>
      
      {/* Latest logs */}
      <div>
        <h4 className="text-sm font-medium text-emerald-300 mb-1 flex items-center">
          <span>Latest Activity</span>
          <span className="ml-1 h-2 w-2 bg-emerald-400 rounded-full animate-pulse"></span>
        </h4>
        <div className="max-h-40 overflow-y-auto bg-gray-900 rounded-md p-2 text-xs">
          {status.log_messages && status.log_messages.length > 0 ? (
            <ul className="space-y-1">
              {status.log_messages.slice(-5).map((log, index) => (
                <li key={index} className="text-gray-300 py-0.5 border-b border-gray-800">
                  {log}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-gray-400">No logs available</p>
          )}
        </div>
      </div>
    </div>
  )
} 