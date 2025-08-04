import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Upload from './pages/Upload';
import Results from './pages/Results';
import Configure from './pages/Configure';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/results" element={<Results />} />
          <Route path="/configure" element={<Configure />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
