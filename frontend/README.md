# RAG Chatbot Frontend

A clean, modern Next.js 15 frontend for the RAG Chatbot API.

## Features

- **Chat Interface** — Ask questions about uploaded documents
- **Source Citations** — Expandable source cards with relevance scores
- **Document Manager** — Upload (drag & drop or click), list, and delete documents
- **Real-time Status** — Health polling shows LLM provider chain, document counts, and system status
- **Responsive Design** — Collapsible sidebar for mobile, full layout for desktop

## Quick Start

### 1. Install dependencies
```bash
cd frontend
npm install
```

### 2. Start the backend (in another terminal)
```bash
cd ..
uvicorn api.main:app --reload
```

### 3. Start the frontend
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Configuration

The frontend proxies API requests to `http://127.0.0.1:8000` via Next.js rewrites in `next.config.js`:

```js
// next.config.js
async rewrites() {
  return [
    {
      source: "/api/:path*",
      destination: "http://127.0.0.1:8000/api/:path*",
    },
  ];
}
```

If your backend runs on a different port or host, update this file.

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx        # Root layout
│   │   ├── page.tsx          # Main page (chat + docs)
│   │   └── globals.css       # Tailwind + custom styles
│   ├── components/
│   │   ├── Header.tsx        # Top bar with status
│   │   ├── ChatPanel.tsx     # Chat messages + input
│   │   ├── DocumentsPanel.tsx # Upload + document list
│   │   └── StatusBar.tsx     # Bottom system status
│   ├── hooks/
│   │   ├── useChat.ts        # Chat state management
│   │   └── useDocuments.ts   # Document CRUD state
│   └── lib/
│       ├── api.ts            # API client functions
│       └── utils.ts          # Tailwind merge helper
├── next.config.js            # API proxy config
├── tailwind.config.ts
└── package.json
```

## API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | System status |
| `/api/chat` | POST | Send query, get answer |
| `/api/documents` | GET | List uploaded documents |
| `/api/documents/upload` | POST | Upload new document |
| `/api/documents/{id}` | DELETE | Delete document |
