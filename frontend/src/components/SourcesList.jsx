import React, { useState } from 'react';
import { BookOpen, ChevronRight } from 'lucide-react';
import SourceModal from './SourceModal';

export default function SourcesList({ sources = [] }) {
  const [selectedSource, setSelectedSource] = useState(null);

  if (!sources || sources.length === 0) return null;

  return (
    <>
      <div className="sources-section">
        <div className="sources-label">
          <BookOpen size={13} color="#888888" />
          <span>Sources ({sources.length})</span>
        </div>

        <div className="sources-pill-row">
          {sources.map((src, idx) => {
            const label = src.breadcrumb ? src.breadcrumb.split(' > ').pop() : `Chunk ${idx + 1}`;
            return (
              <button
                key={idx}
                className="source-chip"
                onClick={() => setSelectedSource(src)}
                title={src.breadcrumb || `Source ${idx + 1}`}
              >
                <span className="source-chip-index">[{idx + 1}]</span>
                <span>{label}</span>
                {src.score !== undefined && (
                  <span className="source-chip-score">{(src.score * 100).toFixed(0)}%</span>
                )}
                <ChevronRight size={12} color="var(--text-muted)" />
              </button>
            );
          })}
        </div>
      </div>

      {selectedSource && (
        <SourceModal source={selectedSource} onClose={() => setSelectedSource(null)} />
      )}
    </>
  );
}
