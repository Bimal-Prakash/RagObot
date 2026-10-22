import React from 'react';

export default function SourcesDrawer({ sources = [] }) {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="glass-panel sources-card" id="sources-drawer">
      <div className="sources-header">
        <span>📄 Retrieved Markdown Semantic Boundaries ({sources.length})</span>
      </div>

      <div className="sources-grid">
        {sources.map((src, idx) => (
          <div className="source-item" key={idx} id={`source-item-${idx}`}>
            <div className="source-breadcrumb">
              {src.breadcrumb || `Section ${idx + 1}`}
            </div>
            {src.score !== undefined && (
              <div className="source-score">
                Similarity Score: <strong>{src.score}</strong>
              </div>
            )}
            <div className="source-snippet">
              {src.content ? src.content.slice(0, 140).trim() + '...' : 'No preview available'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
