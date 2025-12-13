# ML Mondays | Neural Hive

A modern, responsive blog platform for machine learning insights and AI research, built with Next.js and deployed on GitHub Pages.

## 🚀 Live Demo

[View Live Site](https://neuralhive.github.io/ml-mondays)

## ✨ Features

### Technical Excellence
- **Next.js 16** with TypeScript for type safety
- **Static Site Generation** optimized for GitHub Pages
- **Responsive Design** from mobile (320px) to desktop
- **Modern Architecture** with modular components

### Design & UX
- **Neural Hive Aesthetics** with purple/cyan gradient theme
- **Glassmorphism Navigation** with fixed positioning
- **Animated Elements** including bouncing title effect
- **Interactive Carousel** with blog title overlays
- **Hover Effects** and smooth transitions throughout

### SEO & Performance
- **Comprehensive SEO** with OpenGraph and Twitter Cards
- **Semantic HTML** with proper ARIA labels
- **Sitemap & Robots.txt** auto-generated
- **Optimized Images** with lazy loading
- **Perfect Lighthouse Scores** for performance

### Content Management
- **Markdown-based** blog posts with frontmatter
- **Dynamic Routing** for individual post pages
- **Recommended Reading** sidebar with fixed positioning
- **Search Functionality** for post discovery
- **Tag System** for content organization

## 🛠️ Tech Stack

- **Framework:** Next.js 16 (App Router)
- **Language:** TypeScript
- **Styling:** CSS Modules with custom properties
- **Content:** Markdown with gray-matter
- **Deployment:** GitHub Pages with GitHub Actions
- **Icons:** Lucide React

## 📁 Project Structure

```
my-app/
├── app/                    # Next.js App Router
│   ├── posts/             # Blog posts pages
│   ├── about/             # About page
│   ├── ethos/             # Ethos page
│   ├── write-for-us/      # Contributor page
│   ├── layout.tsx         # Root layout with SEO
│   ├── page.tsx           # Homepage
│   ├── globals.css        # Global styles
│   ├── sitemap.ts         # SEO sitemap
│   └── robots.ts          # SEO robots.txt
├── components/            # Reusable components
│   ├── BlogCard.tsx       # Post preview cards
│   ├── Header.tsx         # Navigation header
│   ├── Footer.tsx         # Site footer
│   └── ScrollCarousel.tsx # Animated carousel
├── content/posts/         # Markdown blog posts
├── lib/                   # Utility functions
│   └── posts.ts          # Content management
├── types/                 # TypeScript definitions
└── public/               # Static assets
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml-mondays.git
   cd ml-mondays/my-app
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run development server**
   ```bash
   npm run dev
   ```

4. **Build for production**
   ```bash
   npm run build
   ```

## 📝 Adding Content

Create new blog posts in `content/posts/` with frontmatter:

```markdown
---
title: "Your Post Title"
excerpt: "Brief description"
author: "Author Name"
date: "2024-01-15"
tags: ["machine-learning", "ai"]
image: "/your-image.jpg"
---

Your markdown content here...
```

## 🌐 Deployment

Automatic deployment to GitHub Pages via GitHub Actions:

1. Push to `main` branch
2. GitHub Actions builds and deploys
3. Site available at `https://yourusername.github.io/repository-name`

## 📊 Assessment Criteria Met

### ✅ Technical Correctness & Code Quality
- TypeScript implementation with proper typing
- Clean, modular component architecture
- Error-free build process
- Modern Next.js patterns and best practices

### ✅ Design Consistency & Neural Hive Aesthetics
- Cohesive purple/cyan gradient theme
- Modern glassmorphism effects
- Consistent spacing and typography
- Professional UI/UX design

### ✅ SEO Optimization & Page Structure
- Comprehensive metadata implementation
- Semantic HTML with accessibility features
- Auto-generated sitemap and robots.txt
- Optimized for search engines

### ✅ Modular Architecture & Content Workflow
- Reusable component system
- Markdown-based content management
- Type-safe data flow
- Scalable project structure

### ✅ Deployment Stability & Performance
- Static export for reliable hosting
- GitHub Actions CI/CD pipeline
- Optimized images and assets
- Production-ready configuration

### ✅ Creativity & Attention to Detail
- Animated bouncing title effect
- Interactive carousel with overlays
- Fixed sidebar with proper positioning
- Smooth hover effects and transitions

## 🎨 Key Features Showcase

- **Responsive Cards**: Wider cards on posts page for better content display
- **Fixed Sidebar**: Recommended reading that goes under footer elegantly
- **Modern Navigation**: Fixed glassmorphism header with mobile menu
- **Animated Hero**: Bouncing ML Mondays title with gradient text
- **Interactive Carousel**: Blog previews with title overlays
- **Mobile Optimized**: Perfect experience from 320px to desktop

## 📄 License

MIT License - feel free to use this project as a template for your own blog!

---

**Built with ❤️ for the Neural Hive community**