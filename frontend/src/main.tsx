import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import { Layout } from './components/Layout'
import { Overview } from './pages/Overview'
import { Documents } from './pages/Documents'
import { Query } from './pages/Query'
import { Experiments } from './pages/Experiments'
import { Settings } from './pages/Settings'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="documents" element={<Documents />} />
          <Route path="query" element={<Query />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
