/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  theme: {
    extend: {
      colors: {
        gray: {
          950: '#0C0C0F',
          900: '#141419',
          850: '#1A1A20',
          800: '#202026',
          700: '#2D2D35',
          600: '#4B4B55',
          500: '#68686F',
          400: '#8E8E99',
          300: '#D1D1D6',
          200: '#E5E5EA',
          100: '#F2F2F7',
          50: '#F8F8FC',
          750: '#2F3136',
          850: '#1E1F22',
          950: '#101114',
        },
        emerald: {
          50: '#ECFDF5',
          100: '#D1FAE5',
          200: '#A7F3D0',
          300: '#6EE7B7',
          400: '#34D399',
          500: '#10B981',
          600: '#059669',
          700: '#047857',
          800: '#065F46',
          900: '#064E3B',
          950: '#042f2e',
        },
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.gray.300'),
            a: {
              color: theme('colors.emerald.400'),
              '&:hover': {
                color: theme('colors.emerald.300'),
              },
              textDecoration: 'none',
            },
            'h1, h2, h3, h4, h5, h6': {
              color: theme('colors.white'),
              marginTop: '1.5rem',
              marginBottom: '0.75rem',
            },
            h1: {
              fontSize: '1.5rem',
            },
            h2: {
              fontSize: '1.3rem',
            },
            h3: {
              fontSize: '1.1rem',
            },
            pre: {
              backgroundColor: theme('colors.gray.800'),
              borderRadius: '0.375rem',
              padding: '0.75rem 1rem',
            },
            code: {
              backgroundColor: 'rgba(71, 85, 105, 0.3)',
              color: theme('colors.gray.200'),
              borderRadius: '0.25rem',
              padding: '0.125rem 0.25rem',
              fontWeight: '400',
            },
            'pre code': {
              backgroundColor: 'transparent',
              padding: '0',
            },
            strong: {
              color: theme('colors.white'),
            },
            blockquote: {
              color: theme('colors.gray.300'),
              borderLeftColor: theme('colors.gray.600'),
            },
            'ul, ol': {
              paddingLeft: '1.5rem',
            },
          },
        },
        invert: {
          css: {
            color: theme('colors.gray.300'),
            a: {
              color: theme('colors.emerald.400'),
              '&:hover': {
                color: theme('colors.emerald.300'),
              },
            },
            'h1, h2, h3, h4, h5, h6': {
              color: theme('colors.white'),
            },
            strong: {
              color: theme('colors.white'),
            },
          }
        }
      }),
      animation: {
        fadeIn: 'fadeIn 0.3s ease-in-out',
        fadeSlideIn: 'fadeSlideIn 0.5s ease-out forwards',
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        bounce: 'bounce 1.5s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeSlideIn: {
          '0%': { 
            opacity: '0',
            transform: 'translateY(10px)'
          },
          '100%': { 
            opacity: '1',
            transform: 'translateY(0)'
          },
        },
      },
      boxShadow: {
        subtle: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        card: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}

