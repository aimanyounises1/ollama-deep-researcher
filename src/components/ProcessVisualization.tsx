'use client'

import { StatusType } from '@/types'

interface ProcessVisualizationProps {
  status: StatusType | null
  isVisible: boolean
}

export function ProcessVisualization({ status, isVisible }: ProcessVisualizationProps) {
  if (!isVisible || !status) return null
  
  // Define process steps and their connections
  const steps = [
    { id: 'init', label: 'Initialization', x: 50, y: 20 },
    { id: 'query', label: 'Query Generation', x: 50, y: 80 },
    { id: 'jira', label: 'JIRA Search', x: 20, y: 140 },
    { id: 'confluence', label: 'Confluence Search', x: 50, y: 140 },
    { id: 'perforce', label: 'Perforce Search', x: 80, y: 140 },
    { id: 'analyze', label: 'Result Analysis', x: 50, y: 200 },
    { id: 'summarize', label: 'Generate Summary', x: 50, y: 260 },
    { id: 'complete', label: 'Complete', x: 50, y: 320 }
  ]
  
  // Define connections between steps
  const connections = [
    { from: 'init', to: 'query' },
    { from: 'query', to: 'jira' },
    { from: 'query', to: 'confluence' },
    { from: 'query', to: 'perforce' },
    { from: 'jira', to: 'analyze' },
    { from: 'confluence', to: 'analyze' },
    { from: 'perforce', to: 'analyze' },
    { from: 'analyze', to: 'summarize' },
    { from: 'summarize', to: 'complete' }
  ]
  
  // Determine current step
  const getCurrentStepId = () => {
    const step = status.current_step?.toLowerCase() || '';
    
    if (step.includes('init') || step.includes('start')) return 'init';
    if (step.includes('query') || step.includes('search')) return 'query';
    if (step.includes('jira')) return 'jira';
    if (step.includes('confluence')) return 'confluence';
    if (step.includes('perforce')) return 'perforce';
    if (step.includes('analyz') || step.includes('process')) return 'analyze';
    if (step.includes('summar')) return 'summarize';
    if (step.includes('complet') || step.includes('finish')) return 'complete';
    
    // If cannot determine, use progress percentage
    const progress = status.progress || 0;
    if (progress >= 100) return 'complete';
    if (progress >= 85) return 'summarize';
    if (progress >= 65) return 'analyze';
    if (progress >= 45) return 'jira'; // Just default to one of the search steps
    if (progress >= 25) return 'query';
    return 'init';
  }
  
  const currentStepId = getCurrentStepId();
  
  // Get the relevant CSS classes for a step
  const getStepClasses = (stepId: string) => {
    // If this is the current step
    if (stepId === currentStepId) {
      return 'bg-emerald-600 border-emerald-400 text-white shadow-lg shadow-emerald-900/30 ring-2 ring-emerald-400 scale-110 z-10';
    }
    
    // If this step has already been completed
    const completedSteps = getCompletedSteps();
    if (completedSteps.includes(stepId)) {
      return 'bg-emerald-800 border-emerald-700 text-white';
    }
    
    // Future steps
    return 'bg-gray-700 border-gray-600 text-gray-300';
  }
  
  // Get connection classes
  const getConnectionClasses = (from: string, to: string) => {
    const completedSteps = getCompletedSteps();
    
    if (completedSteps.includes(from) && completedSteps.includes(to)) {
      return 'stroke-emerald-500 stroke-2';
    }
    
    if (from === currentStepId || (completedSteps.includes(from) && to === currentStepId)) {
      return 'stroke-emerald-400 stroke-2 stroke-dashed';
    }
    
    return 'stroke-gray-600 stroke-1';
  }
  
  // Get completed steps based on current step
  const getCompletedSteps = () => {
    const orderedSteps = ['init', 'query', 'jira', 'confluence', 'perforce', 'analyze', 'summarize', 'complete'];
    const currentIndex = orderedSteps.indexOf(currentStepId);
    
    // Special case for parallel steps (jira, confluence, perforce)
    const parallelSteps = ['jira', 'confluence', 'perforce'];
    
    if (parallelSteps.includes(currentStepId)) {
      // Only return the steps before parallel steps, plus the current one
      return [...orderedSteps.slice(0, orderedSteps.indexOf('jira')), currentStepId];
    } else if (currentIndex > orderedSteps.indexOf('perforce')) {
      // If we're past the parallel steps, include all three
      return [...orderedSteps.slice(0, currentIndex), ...parallelSteps.filter(s => s !== currentStepId)];
    } else {
      // Otherwise just return all steps up to current one
      return orderedSteps.slice(0, currentIndex);
    }
  }
  
  return (
    <div className="mt-4 mb-6 p-4 bg-gray-800 rounded-lg border border-gray-700">
      <h3 className="text-md font-medium text-white mb-4">Research Process Visualization</h3>
      
      <div className="relative w-full h-[350px]">
        {/* Connection lines */}
        <svg className="absolute inset-0 w-full h-full z-0">
          {connections.map((conn, i) => {
            const fromStep = steps.find(s => s.id === conn.from);
            const toStep = steps.find(s => s.id === conn.to);
            
            if (!fromStep || !toStep) return null;
            
            return (
              <line 
                key={i}
                x1={`${fromStep.x}%`} 
                y1={`${fromStep.y}px`} 
                x2={`${toStep.x}%`} 
                y2={`${toStep.y}px`}
                className={`${getConnectionClasses(conn.from, conn.to)} transition-all duration-500`}
                strokeLinecap="round"
              />
            );
          })}
        </svg>
        
        {/* Process steps */}
        {steps.map(step => (
          <div
            key={step.id}
            className={`absolute rounded-lg border px-2 py-1 text-xs font-medium transform -translate-x-1/2 transition-all duration-500 ${getStepClasses(step.id)}`}
            style={{ 
              left: `${step.x}%`, 
              top: `${step.y}px`,
              minWidth: '100px',
              textAlign: 'center'
            }}
          >
            {step.label}
            {step.id === currentStepId && (
              <div className="absolute -top-1 -right-1 h-3 w-3 bg-emerald-400 rounded-full animate-ping"></div>
            )}
          </div>
        ))}
      </div>
      
      <div className="mt-3 text-center text-xs text-gray-400">
        Current step: <span className="text-emerald-400 font-medium">{status.current_step || 'Processing...'}</span>
      </div>
    </div>
  )
} 