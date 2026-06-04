import ReactDOM from 'react-dom/client';
import { StrictMode } from 'react';
import App from './App';
import './style.css';
import './design-tokens.css';

ReactDOM.createRoot(document.getElementById('app')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
