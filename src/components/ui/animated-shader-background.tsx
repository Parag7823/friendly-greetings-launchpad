import { useEffect, useRef } from 'react';

export const AnimatedShaderBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Animated lines configuration
    const lines: Array<{
      x: number;
      y: number;
      length: number;
      angle: number;
      speed: number;
      color: string;
      opacity: number;
    }> = [];

    // Create initial lines - more lines with brighter colors
    for (let i = 0; i < 80; i++) {
      const colorChoice = Math.random();
      let color;
      if (colorChoice < 0.33) {
        color = '#3b82f6'; // Bright Blue
      } else if (colorChoice < 0.66) {
        color = '#8b5cf6'; // Bright Purple
      } else {
        color = '#06b6d4'; // Cyan
      }
      
      lines.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        length: Math.random() * 250 + 150,
        angle: Math.random() * Math.PI * 2,
        speed: Math.random() * 0.8 + 0.3,
        color: color,
        opacity: Math.random() * 0.6 + 0.4
      });
    }

    // Animation loop
    let animationId: number;
    const animate = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.03)'; // Lighter fade for more visible trails
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      lines.forEach((line) => {
        // Update position
        line.x += Math.cos(line.angle) * line.speed;
        line.y += Math.sin(line.angle) * line.speed;

        // Wrap around screen
        if (line.x < 0) line.x = canvas.width;
        if (line.x > canvas.width) line.x = 0;
        if (line.y < 0) line.y = canvas.height;
        if (line.y > canvas.height) line.y = 0;

        // Draw line with glow effect
        ctx.beginPath();
        ctx.moveTo(line.x, line.y);
        ctx.lineTo(
          line.x + Math.cos(line.angle) * line.length,
          line.y + Math.sin(line.angle) * line.length
        );
        
        // Add glow
        ctx.shadowBlur = 10;
        ctx.shadowColor = line.color;
        ctx.strokeStyle = line.color;
        ctx.globalAlpha = line.opacity;
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Reset shadow
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
      });

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full -z-10"
      style={{ background: '#000000' }}
    />
  );
};
