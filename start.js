import { app, PORT, HF_API_KEY } from './server.js';

app.listen(PORT, () => {
  console.log(`\nMultimodal AI Detector running at http://localhost:${PORT}\n`);
  if (!HF_API_KEY || HF_API_KEY === 'your_huggingface_api_token_here') {
    console.warn('WARNING: HF_API_KEY is not set. API calls will fail.\n');
  }
});