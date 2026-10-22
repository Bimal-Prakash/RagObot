import React from 'react';
import { X, FileText, CheckCircle2 } from 'lucide-react';

export default function SourceModal({ source, onClose }) {
  if (!source) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <div className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileText size={16} color="var(--accent-primary)" />
              <span>{source.breadcrumb || 'Retrieved Knowledge Chunk'}</span>
            </div>
            {source.score !== undefined && (
              <div className="modal-subtitle" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={12} color="var(--accent-emerald)" />
                <span>Semantic Relevance Score: {source.score}</span>
              </div>
            )}
          </div>
          <button className="modal-close-btn" onClick={onClose} title="Close preview">
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {source.content || 'No chunk content available.'}
        </div>
      </div>
    </div>
  );
}
