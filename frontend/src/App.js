import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Pages
import HomePage from './pages/HomePage';
import ResearchPage from './pages/ResearchPage';
import QuizPage from './pages/QuizPage';
import ChatPageTailwind from './pages/ChatPageTailwind';

// Components
import Header from './components/Header';
import Footer from './components/Footer';

function App() {
  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-white">
      <Header />
      <main className="flex-1 container mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<ChatPageTailwind />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/quiz" element={<QuizPage />} />
          <Route path="/chat" element={<Navigate to="/" replace />} />
          <Route path="/chatgpt" element={<ChatPageTailwind />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App; 