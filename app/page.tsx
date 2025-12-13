import Link from 'next/link';
import { ArrowRight, Brain, Zap } from 'lucide-react';
import { getAllPosts } from '@/lib/posts';
import BlogCard from '@/components/BlogCard';
import ScrollCarousel from '@/components/ScrollCarousel';

export default function Home() {
  // ML Mondays - Neural Hive Blog Platform
  const posts = getAllPosts();
  const recentPosts = posts.slice(0, 6);

  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <header>
            <h1 className="hero-title">
              <span className="gradient-text animate-bounce">ML Mondays</span>
            </h1>
          </header>
          <ScrollCarousel posts={posts} />
          <p className="hero-description">
            Exploring the frontiers of machine learning, AI research, and neural networks. 
            Weekly insights from the Neural Hive community.
          </p>
          <nav className="hero-buttons" aria-label="Main navigation">
            <Link href="/posts" className="btn-primary">
              Explore Posts <ArrowRight className="btn-icon" size={20} />
            </Link>
            <Link href="/write-for-us" className="btn-secondary">
              Write for Us
            </Link>
          </nav>
        </div>
      </section>

      {/* Features */}
      <section className="features-section">
        <div className="features-grid">
          <div className="feature-card">
            <Brain className="feature-icon purple" size={48} />
            <h3 className="feature-title">Deep Learning Insights</h3>
            <p className="feature-description">Cutting-edge research and practical applications in neural networks</p>
          </div>
          <div className="feature-card">
            <Zap className="feature-icon cyan" size={48} />
            <h3 className="feature-title">Weekly Updates</h3>
            <p className="feature-description">Fresh content every Monday covering the latest in ML and AI</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon-gradient"></div>
            <h3 className="feature-title">Community Driven</h3>
            <p className="feature-description">Written by researchers, engineers, and ML enthusiasts</p>
          </div>
        </div>
      </section>

      {/* Recent Posts */}
      <section className="recent-section" aria-labelledby="latest-posts">
        <div className="section-header">
          <h2 id="latest-posts" className="section-title">Latest Posts</h2>
          <Link href="/posts" className="view-all-link" aria-label="View all blog posts">
            View all posts →
          </Link>
        </div>
        <div className="recent-grid" role="list">
          {recentPosts.map((post) => (
            <div key={post.slug} role="listitem">
              <BlogCard post={post} />
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}