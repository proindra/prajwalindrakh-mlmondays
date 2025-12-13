import { Mail, FileText, Users, CheckCircle } from 'lucide-react';

export default function WriteForUsPage() {
  return (
    <div className="page-container">
      <div className="page-hero">
        <h1 className="page-title">
          Write for <span className="gradient-text">Neural Hive</span>
        </h1>
        <p className="page-subtitle">
          Share your machine learning expertise with our growing community of researchers, 
          engineers, and enthusiasts worldwide.
        </p>
      </div>

      <div className="write-benefits">
        <div className="benefit-card">
          <Users className="benefit-icon purple" size={32} />
          <h3>Join Our Community</h3>
          <p>
            Connect with like-minded ML practitioners and contribute to the advancement 
            of machine learning knowledge.
          </p>
        </div>
        <div className="benefit-card">
          <FileText className="benefit-icon cyan" size={32} />
          <h3>Share Your Expertise</h3>
          <p>
            Whether you're a researcher, engineer, or student, your unique perspective 
            can help others learn and grow.
          </p>
        </div>
      </div>

      <div className="content-card">
        <h2>What We're Looking For</h2>
        <div className="looking-grid">
          <div>
            <h4 className="looking-title purple">Technical Content</h4>
            <ul>
              <li>• Research paper breakdowns and analysis</li>
              <li>• Implementation tutorials and guides</li>
              <li>• Algorithm explanations and comparisons</li>
              <li>• Production ML case studies</li>
              <li>• Tool reviews and comparisons</li>
            </ul>
          </div>
          <div>
            <h4 className="looking-title cyan">Industry Insights</h4>
            <ul>
              <li>• Career advice and growth paths</li>
              <li>• Interview preparation guides</li>
              <li>• Industry trend analysis</li>
              <li>• Startup and product stories</li>
              <li>• Conference and event summaries</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="content-card">
        <h2>Submission Process</h2>
        <div className="process-steps">
          <div className="process-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Pitch Your Idea</h4>
              <p>
                Send us a brief outline (200-300 words) of your proposed article. Include the main topic, 
                key points you'll cover, and why it's valuable to our community.
              </p>
            </div>
          </div>
          <div className="process-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Get Approved</h4>
              <p>
                Our editorial team will review your pitch within 3-5 business days. We'll provide 
                feedback and guidance to help shape your article.
              </p>
            </div>
          </div>
          <div className="process-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Write & Submit</h4>
              <p>
                Create your article following our style guide. Submit as a markdown file with 
                any accompanying code, images, or datasets.
              </p>
            </div>
          </div>
          <div className="process-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Review & Publish</h4>
              <p>
                We'll review your article for technical accuracy and clarity, then work with you 
                on any revisions before publishing.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="writer-benefits">
        <h2>Writer Benefits</h2>
        <div className="benefits-grid">
          <div className="benefit-item">
            <CheckCircle className="check-icon" size={20} />
            <span>Byline and author bio</span>
          </div>
          <div className="benefit-item">
            <CheckCircle className="check-icon" size={20} />
            <span>Social media promotion</span>
          </div>
          <div className="benefit-item">
            <CheckCircle className="check-icon" size={20} />
            <span>Portfolio piece for your career</span>
          </div>
          <div className="benefit-item">
            <CheckCircle className="check-icon" size={20} />
            <span>Access to our writer community</span>
          </div>
          <div className="benefit-item">
            <CheckCircle className="check-icon" size={20} />
            <span>Editorial support and feedback</span>
          </div>
          <div className="benefit-item">
            <CheckCircle className="check-icon" size={20} />
            <span>Networking opportunities</span>
          </div>
        </div>
      </div>

      <div className="page-cta">
        <h2>Ready to Get Started?</h2>
        <p>Have an idea for ML Mondays? We'd love to hear from you!</p>
        <div className="cta-buttons">
          <a
            href="mailto:writers@neuralhive.dev?subject=ML Mondays Pitch"
            className="btn-primary neural-glow"
          >
            <Mail size={20} />
            Send Your Pitch
          </a>
          <a href="/ethos/" className="btn-secondary">
            Read Our Guidelines
          </a>
        </div>
        <p className="contact-info">
          Email: writers@neuralhive.dev | Response time: 3-5 business days
        </p>
      </div>
    </div>
  );
}