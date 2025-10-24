import React from 'react';

interface MarkdownMessageProps {
  content: string;
  className?: string;
}

export const MarkdownMessage: React.FC<MarkdownMessageProps> = ({ content, className = '' }) => {
  // Simple markdown parser for common patterns
  const parseMarkdown = (text: string): JSX.Element[] => {
    const lines = text.split('\n');
    const elements: JSX.Element[] = [];
    let key = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Skip empty lines
      if (!line.trim()) {
        elements.push(<br key={`br-${key++}`} />);
        continue;
      }

      // Headers (##)
      if (line.startsWith('## ')) {
        elements.push(
          <h3 key={`h3-${key++}`} className="font-bold text-sm mt-3 mb-1">
            {line.replace('## ', '')}
          </h3>
        );
        continue;
      }

      // Headers (#)
      if (line.startsWith('# ')) {
        elements.push(
          <h2 key={`h2-${key++}`} className="font-bold text-base mt-3 mb-1">
            {line.replace('# ', '')}
          </h2>
        );
        continue;
      }

      // Bullet points (-, •, ✓)
      if (line.match(/^[\s]*[-•✓→]\s/)) {
        const content = line.replace(/^[\s]*[-•✓→]\s/, '');
        elements.push(
          <div key={`bullet-${key++}`} className="flex gap-2 ml-2 my-1">
            <span className="text-primary">•</span>
            <span>{formatInlineMarkdown(content)}</span>
          </div>
        );
        continue;
      }

      // Numbered lists
      if (line.match(/^[\s]*\d+\.\s/)) {
        const content = line.replace(/^[\s]*\d+\.\s/, '');
        const number = line.match(/^[\s]*(\d+)\./)?.[1];
        elements.push(
          <div key={`numbered-${key++}`} className="flex gap-2 ml-2 my-1">
            <span className="text-primary font-semibold">{number}.</span>
            <span>{formatInlineMarkdown(content)}</span>
          </div>
        );
        continue;
      }

      // Emoji headers (🎯, 💡, etc.)
      if (line.match(/^[🎯💡📊✅⚠️🚀💰📈]/)) {
        elements.push(
          <p key={`emoji-${key++}`} className="font-semibold my-2">
            {formatInlineMarkdown(line)}
          </p>
        );
        continue;
      }

      // Regular paragraphs
      elements.push(
        <p key={`p-${key++}`} className="my-1">
          {formatInlineMarkdown(line)}
        </p>
      );
    }

    return elements;
  };

  // Format inline markdown (bold, italic, code)
  const formatInlineMarkdown = (text: string): React.ReactNode => {
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let key = 0;

    while (remaining.length > 0) {
      // Bold (**text**)
      const boldMatch = remaining.match(/\*\*([^*]+)\*\*/);
      if (boldMatch) {
        const before = remaining.substring(0, boldMatch.index);
        if (before) parts.push(<span key={`text-${key++}`}>{before}</span>);
        parts.push(<strong key={`bold-${key++}`} className="font-bold">{boldMatch[1]}</strong>);
        remaining = remaining.substring((boldMatch.index || 0) + boldMatch[0].length);
        continue;
      }

      // Code (`code`)
      const codeMatch = remaining.match(/`([^`]+)`/);
      if (codeMatch) {
        const before = remaining.substring(0, codeMatch.index);
        if (before) parts.push(<span key={`text-${key++}`}>{before}</span>);
        parts.push(
          <code key={`code-${key++}`} className="bg-white/10 px-1 py-0.5 rounded text-xs">
            {codeMatch[1]}
          </code>
        );
        remaining = remaining.substring((codeMatch.index || 0) + codeMatch[0].length);
        continue;
      }

      // No more special formatting
      parts.push(<span key={`text-${key++}`}>{remaining}</span>);
      break;
    }

    return parts.length > 0 ? <>{parts}</> : text;
  };

  return (
    <div className={`markdown-content text-xs leading-relaxed ${className}`}>
      {parseMarkdown(content)}
    </div>
  );
};
