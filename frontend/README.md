# PHI Anonymization Frontend

React-based Single Page Application (SPA) for the PHI Anonymization tool.

## Features

- Modern, responsive UI with drag-and-drop file upload
- LLM configuration with localStorage persistence
- Multiple processing modes (auto-detect, vision, OCR)
- Real-time processing feedback
- Secure file download
- GitHub Pages ready

## Development

### Prerequisites

- Node.js 16+
- Backend server running at `http://localhost:8000`

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Visit `http://localhost:5173` in your browser.

### Build for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

### Deploy to GitHub Pages

```bash
npm run deploy
```

This will build the app and deploy it to the `gh-pages` branch.

## Configuration

### Backend URL

The backend URL can be configured in the UI. It defaults to `http://localhost:8000` and is stored in localStorage.

### Vite Configuration

Edit `vite.config.js` to change:
- `base`: Base public path (for GitHub Pages deployment)
- `server.port`: Development server port
- `server.proxy`: Proxy configuration for API requests

## Project Structure

```
src/
├── components/
│   ├── ConfigForm.jsx        # LLM configuration form
│   ├── ConfigForm.css
│   ├── FileUpload.jsx        # File upload with drag-and-drop
│   └── FileUpload.css
├── App.jsx                   # Main application component
├── App.css                   # Global styles
└── main.jsx                  # Application entry point
```

## Components

### App.jsx

Main application component that:
- Manages configuration state
- Handles backend URL configuration
- Conditionally renders ConfigForm or FileUpload

### ConfigForm.jsx

LLM configuration form that:
- Collects Azure OpenAI credentials
- Persists settings to localStorage
- Submits configuration to backend

### FileUpload.jsx

File upload component that:
- Supports drag-and-drop
- Allows mode selection (auto/vision/ocr)
- Shows processing status
- Handles file download

## Styling

This project uses plain CSS (no framework) with:
- CSS Grid and Flexbox for layout
- CSS custom properties for theming
- Responsive design with media queries
- Smooth transitions and animations

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions

## API Integration

The frontend communicates with the backend at the configured URL:

- `POST /api/config` - Set LLM configuration
- `GET /api/config/status` - Check configuration status
- `POST /api/process` - Process a file
- `GET /api/download/{job_id}/{filename}` - Download result
- `DELETE /api/cleanup/{job_id}` - Clean up temporary files

## Deployment

### GitHub Pages

1. Update repository name in `vite.config.js` if needed:
   ```js
   base: '/your-repo-name/'
   ```

2. Deploy:
   ```bash
   npm run deploy
   ```

### Other Platforms

**Netlify/Vercel:**
- Build command: `npm run build`
- Publish directory: `dist`

**Static Hosting (S3, Firebase, etc.):**
- Upload contents of `dist/` directory

## Troubleshooting

### CORS Errors

Ensure the backend CORS configuration includes your frontend origin.

### API Not Found

Check that:
1. Backend is running
2. Backend URL is correct in the UI
3. Proxy configuration in Vite is correct

### Build Fails

Try:
```bash
rm -rf node_modules dist
npm install
npm run build
```

## License

[Your License Here]
