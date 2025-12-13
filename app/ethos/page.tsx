import { BookOpen, Heart, Zap, Shield } from 'lucide-react';

export default function EthosPage() {
  return (
    <div className="page-container">
      <div className="page-hero">
        <h1 className="page-title">
          Our <span className="gradient-text">Ethos</span>
        </h1>
        <p className="page-subtitle">
          The principles that guide everything we create at Neural Hive
        </p>
      </div>

      <div className="ethos-content">
        <div className="ethos-principle">
          <div className="principle-header">
            <BookOpen className="principle-icon purple" size={32} />
            <h2>Quality Over Quantity</h2>
          </div>
          <p>
            We believe in deep, thoughtful content rather than quick takes. Every ML Mondays post 
            undergoes rigorous review to ensure accuracy, clarity, and practical value.
          </p>
          <ul>
            <li>• Peer-reviewed by domain experts</li>
            <li>• Fact-checked against primary sources</li>
            <li>• Tested code examples and implementations</li>
            <li>• Clear explanations accessible to various skill levels</li>
          </ul>
        </div>

        <div className="ethos-principle">
          <div className="principle-header">
            <Heart className="principle-icon red" size={32} />
            <h2>Community-Centric</h2>
          </div>
          <p>
            Our content serves the community's needs, not corporate interests. We prioritize 
            educational value and practical applicability over promotional content.
          </p>
          <ul>
            <li>• No hidden agendas or sponsored content without disclosure</li>
            <li>• Open source examples and reproducible research</li>
            <li>• Diverse perspectives from various backgrounds</li>
            <li>• Constructive feedback and collaborative improvement</li>
          </ul>
        </div>

        <div className="ethos-principle">
          <div className="principle-header">
            <Zap className="principle-icon yellow" size={32} />
            <h2>Cutting-Edge Relevance</h2>
          </div>
          <p>
            We stay at the forefront of ML research while maintaining practical grounding. 
            Our content bridges the gap between academic breakthroughs and real-world applications.
          </p>
          <ul>
            <li>• Latest research papers explained in accessible language</li>
            <li>• Production-ready implementations and best practices</li>
            <li>• Industry trends and emerging technologies</li>
            <li>• Balanced coverage of hype vs. reality</li>
          </ul>
        </div>

        <div className="ethos-principle">
          <div className="principle-header">
            <Shield className="principle-icon green" size={32} />
            <h2>Ethical AI Advocacy</h2>
          </div>
          <p>
            We're committed to promoting responsible AI development and deployment. 
            Every piece of content considers the broader implications of the technology we discuss.
          </p>
          <ul>
            <li>• Bias detection and mitigation strategies</li>
            <li>• Privacy-preserving techniques and considerations</li>
            <li>• Environmental impact awareness</li>
            <li>• Inclusive and accessible AI development</li>
          </ul>
        </div>

        <div className="writing-guidelines">
          <h2>Writing Guidelines</h2>
          <div className="guidelines-grid">
            <div>
              <h3 className="guideline-title purple">Content Standards</h3>
              <ul>
                <li>• Original research or unique perspectives</li>
                <li>• Proper attribution and citations</li>
                <li>• Code examples that actually work</li>
                <li>• Clear problem statements and solutions</li>
              </ul>
            </div>
            <div>
              <h3 className="guideline-title cyan">Style Guide</h3>
              <ul>
                <li>• Conversational yet professional tone</li>
                <li>• Technical accuracy without jargon overload</li>
                <li>• Visual aids and diagrams when helpful</li>
                <li>• Actionable takeaways for readers</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="page-cta">
        <h2>Ready to Contribute?</h2>
        <p>Join our community of writers who share these values</p>
        <a href="/write-for-us" className="btn-primary neural-glow">
          Write for Us
        </a>
      </div>
    </div>
  );
}