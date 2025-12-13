import { Users, Target, Lightbulb } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="page-container">
      <div className="page-hero">
        <h1 className="page-title">
          About <span className="gradient-text">Neural Hive</span>
        </h1>
        <p className="page-subtitle">
          We're a community of researchers, engineers, and enthusiasts pushing the boundaries 
          of machine learning and artificial intelligence.
        </p>
      </div>

      <div className="prose-container">
        <div className="content-card">
          <h2>Our Story</h2>
          <p>
            Neural Hive was born from a simple observation: the field of machine learning moves incredibly fast, 
            but quality educational content often lags behind. We saw brilliant researchers making groundbreaking 
            discoveries, but their insights remained locked in academic papers or scattered across social media.
          </p>
          <p>
            ML Mondays represents our commitment to bridging this gap. Every week, we curate and create content 
            that makes cutting-edge ML research accessible, practical, and actionable for everyone from students 
            to senior engineers.
          </p>
        </div>

        <div className="values-grid">
          <div className="value-card">
            <Users className="value-icon" size={48} />
            <h3>Community First</h3>
            <p>Built by the community, for the community. Every article reflects real-world experience.</p>
          </div>
          <div className="value-card">
            <Target className="value-icon cyan" size={48} />
            <h3>Practical Focus</h3>
            <p>We don't just explain theory—we show you how to apply it in production systems.</p>
          </div>
          <div className="value-card">
            <Lightbulb className="value-icon green" size={48} />
            <h3>Innovation</h3>
            <p>Covering the latest breakthroughs before they become mainstream.</p>
          </div>
        </div>

        <div className="content-card">
          <h2>What We Cover</h2>
          <div className="coverage-grid">
            <div>
              <h4 className="coverage-title purple">Research & Theory</h4>
              <ul>
                <li>• Latest paper breakdowns</li>
                <li>• Algorithm deep-dives</li>
                <li>• Mathematical foundations</li>
              </ul>
            </div>
            <div>
              <h4 className="coverage-title cyan">Practical Applications</h4>
              <ul>
                <li>• Implementation tutorials</li>
                <li>• Production case studies</li>
                <li>• Tool comparisons</li>
              </ul>
            </div>
            <div>
              <h4 className="coverage-title green">Industry Insights</h4>
              <ul>
                <li>• Career guidance</li>
                <li>• Interview preparation</li>
                <li>• Market trends</li>
              </ul>
            </div>
            <div>
              <h4 className="coverage-title yellow">Community</h4>
              <ul>
                <li>• Guest contributions</li>
                <li>• Discussion forums</li>
                <li>• Collaborative projects</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="page-cta">
        <h2>Join Our Mission</h2>
        <p>Ready to contribute to the future of ML education?</p>
        <a href="/write-for-us" className="btn-primary neural-glow">
          Write for Us
        </a>
      </div>
    </div>
  );
}