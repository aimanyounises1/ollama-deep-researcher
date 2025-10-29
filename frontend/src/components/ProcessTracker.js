import React from 'react';

function ProcessTracker({ status, currentStep, progress, logMessages = [] }) {
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      case 'processing':
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };
  
  const getStatusIcon = (step, current) => {
    if (status === 'error' && step === current) {
      return (
        <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
        </svg>
      );
    } else if (status === 'completed' && step === current) {
      return (
        <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
        </svg>
      );
    } else if (step === current) {
      return (
        <svg className="w-5 h-5 text-blue-500 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
        </svg>
      );
    } else {
      return (
        <svg className="w-5 h-5 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
        </svg>
      );
    }
  };
  
  const isStepActive = (step) => step === currentStep;
  
  const isStepDone = (index, steps) => {
    if (status === 'completed') return true;
    const currentIndex = steps.indexOf(currentStep);
    return index < currentIndex;
  };
  
  // Define process steps based on the type of task
  const processSteps = currentStep.includes('quiz') ? [
    'initializing',
    'loading_test_data',
    'running_qa',
    'completed'
  ] : [
    'initializing',
    'running_graph', 
    'generating_report',
    'completed'
  ];
  
  // Get most recent log messages (last 5)
  const recentLogs = [...logMessages].slice(-5);
  
  return (
    <div className="bg-gray-800 p-6 mb-6 rounded-lg shadow-md border border-gray-700">
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold text-white">
            Process Status
          </h2>
          <span 
            className={`px-2 py-1 text-xs font-bold rounded-full ${
              status === 'completed' ? 'bg-green-900 text-green-200' : 
              status === 'error' ? 'bg-red-900 text-red-200' : 
              'bg-blue-900 text-blue-200'
            }`}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2.5">
          <div 
            className={`h-2.5 rounded-full ${getStatusColor(status)}`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <p className="text-right text-sm text-gray-400 mt-1">
          {progress}% Complete
        </p>
      </div>
      
      <div className="my-4 border-t border-gray-700"></div>
      
      <h2 className="text-lg font-semibold text-white mb-2">
        Process Steps
      </h2>
      <ul className="space-y-2">
        {processSteps.map((step, index) => (
          <li
            key={step}
            className={`flex items-center p-2 rounded ${
              isStepActive(step) ? 'bg-blue-900/30 border border-blue-500' : ''
            }`}
          >
            <div className="mr-3">
              {isStepDone(index, processSteps) ? (
                <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              ) : (
                getStatusIcon(step, currentStep)
              )}
            </div>
            <span className={`${isStepActive(step) ? 'font-bold text-blue-400' : 'text-gray-300'}`}>
              {step.replace(/_/g, ' ').charAt(0).toUpperCase() + step.replace(/_/g, ' ').slice(1)}
            </span>
          </li>
        ))}
      </ul>
      
      {recentLogs.length > 0 && (
        <>
          <div className="my-4 border-t border-gray-700"></div>
          <h2 className="text-lg font-semibold text-white mb-2">
            Recent Activity
          </h2>
          <ul className="bg-gray-900 rounded border border-gray-700 divide-y divide-gray-700">
            {recentLogs.map((log, index) => (
              <li key={index} className="px-3 py-2">
                <p className="text-sm font-mono text-gray-300">
                  {log}
                </p>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default ProcessTracker; 