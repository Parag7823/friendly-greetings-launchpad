# Chat Interface Layout Fixes

## Issues Fixed

### 1. **Data Sources Panel Overlap** ✅
**Problem**: When Data Sources sidebar opens, it covers the chat interface
**Solution**: Wrap chat in `motion.div` that shifts left by 500px when panel opens

### 2. **Oversized StarBorder Animation** ✅  
**Problem**: Large animated star components dominate the chat input area
**Solution**: Replace with minimal border-only animation on hover

## Implementation

### Changes to `ChatInterface.tsx`:

1. **Add motion import**:
```tsx
import { motion } from 'framer-motion';
```

2. **Remove StarBorder import**:
```tsx
// Remove: import { StarBorder } from './ui/star-border';
```

3. **Wrap chat interface** (around line 353):
```tsx
return (
  <div className="h-full flex bg-background">
    {/* Main Chat Area - Responsive to sidebar */}
    <motion.div 
      className="flex-1 flex flex-col min-w-0"
      animate={{ 
        marginRight: showDataSources ? '500px' : '0px' 
      }}
      transition={{ type: 'spring', damping: 25, stiffness: 200 }}
    >
      {/* All existing chat content */}
      ...
    </motion.div>
    
    {/* Data Sources Panel */}
    <DataSourcesPanel 
      isOpen={showDataSources} 
      onClose={() => setShowDataSources(false)} 
    />
  </div>
);
```

4. **Replace StarBorder input** (around line 436):
```tsx
{/* Chat Input Area - Minimal border animation */}
<div className="border-t border-border p-4 bg-background">
  <div className="max-w-4xl mx-auto">
    <div className="relative rounded-lg border border-border bg-background overflow-hidden group">
      {/* Minimal animated border effect */}
      <div className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <div className="absolute inset-0 rounded-lg border-2 border-primary/20 animate-pulse" />
      </div>
      
      <div className="relative">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Ask anything about your financial data..."
          className="w-full bg-transparent border-none px-4 py-3 pr-12 text-sm text-foreground placeholder-muted-foreground focus:outline-none"
        />
        
        <button
          onClick={handleSendMessage}
          disabled={!message.trim()}
          className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 bg-primary text-primary-foreground rounded-md flex items-center justify-center transition-all duration-200 hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  </div>
</div>
```

## Result

✅ **Responsive Layout**: Chat interface smoothly shifts left when Data Sources opens
✅ **No Overlap**: Both panels visible simultaneously  
✅ **Smooth Animation**: Spring-based transition (damping: 25, stiffness: 200)
✅ **Minimal Border**: Subtle pulse animation on hover only
✅ **Professional UX**: Clean, modern interface without visual clutter

## Testing

1. Open Data Sources panel → Chat should shift left smoothly
2. Close Data Sources panel → Chat should expand back to full width
3. Hover over input → Minimal border pulse animation
4. No layout jumps or abrupt changes
