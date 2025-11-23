import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			fontFamily: {
				'sans': ['Inter', 'system-ui', 'sans-serif'],
				'inter': ['Inter', 'system-ui', 'sans-serif'],
				'mono': ['JetBrains Mono', 'monospace'],
			},
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				// Finley AI Custom Colors
				'finley-accent': 'hsl(var(--finley-accent))',
				'finley-light-accent': 'hsl(var(--finley-light-accent))',
				
				// Neutral/Muted Tones for UI Hierarchy
				'slate': {
					'400': 'hsl(var(--slate-400))',
					'500': 'hsl(var(--slate-500))',
					'700': 'hsl(var(--slate-700))',
					'800': 'hsl(var(--slate-800))',
				},
				
				// Secondary Accent - Steel Blue
				'steel-blue': 'hsl(var(--secondary-accent))',
				
				// Accessibility: Copper Variants
				'copper': {
					'light': 'hsl(var(--copper-light))',
					'dark': 'hsl(var(--copper-dark))',
				},
				
				// Data Visualization Palette
				'chart': {
					'copper': 'hsl(var(--chart-copper))',
					'teal': 'hsl(var(--chart-teal))',
					'purple': 'hsl(var(--chart-purple))',
					'orange': 'hsl(var(--chart-orange))',
					'pink': 'hsl(var(--chart-pink))',
					'cyan': 'hsl(var(--chart-cyan))',
				},
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'star-movement-bottom': {
					'0%': { transform: 'translate(0%, 0%)', opacity: '1' },
					'100%': { transform: 'translate(-100%, 0%)', opacity: '0' },
				},
				'star-movement-top': {
					'0%': { transform: 'translate(0%, 0%)', opacity: '1' },
					'100%': { transform: 'translate(100%, 0%)', opacity: '0' },
				},
				'border-slide': {
					'0%': { backgroundPosition: '200% 0' },
					'100%': { backgroundPosition: '-200% 0' },
				},
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'star-movement-bottom': 'star-movement-bottom linear infinite alternate',
				'star-movement-top': 'star-movement-top linear infinite alternate',
				'border-slide': 'border-slide linear infinite',
			}
		}
	},
	plugins: [
		require("tailwindcss-animate"),
		function ({ addUtilities }) {
			addUtilities({
				'.scrollbar-thin': {
					'scrollbar-width': 'thin',
					'scrollbar-color': 'rgba(160, 160, 160, 0.2) transparent',
				},
				'.scrollbar-thin:hover': {
					'scrollbar-color': 'rgba(160, 160, 160, 0.3) transparent',
				},
				// Webkit scrollbar styles for better cross-browser support
				'.scrollbar-thin::-webkit-scrollbar': {
					'width': '6px',
				},
				'.scrollbar-thin::-webkit-scrollbar-track': {
					'background': 'transparent',
				},
				'.scrollbar-thin::-webkit-scrollbar-thumb': {
					'background-color': 'rgba(160, 160, 160, 0.2)',
					'border-radius': '3px',
				},
				'.scrollbar-thin::-webkit-scrollbar-thumb:hover': {
					'background-color': 'rgba(160, 160, 160, 0.3)',
				},
			})
		}
	],
} satisfies Config;
