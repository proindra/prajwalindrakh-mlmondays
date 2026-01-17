import Link from 'next/link';
import { getImagePath } from '@/lib/utils';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-grid">
          <div className="footer-brand">
            <div className="footer-logo">
              <img src={getImagePath("/neural-hive-logo.svg")} alt="Neural Hive Logo" className="footer-logo-image" />
              <span className="footer-logo-text gradient-text">Neural Insights</span>
            </div>
            <p className="footer-description">
              Exploring the frontiers of machine learning, AI research, and neural networks. 
              Weekly insights from the Neural Hive community.
            </p>
          </div>
          
          <div className="footer-section">
            <h3 className="footer-section-title">Navigation</h3>
            <ul className="footer-links">
              <li><Link href="/" className="footer-link">Home</Link></li>
              <li><Link href="/posts" className="footer-link">Posts</Link></li>
              <li><Link href="/about" className="footer-link">About</Link></li>
              <li><Link href="/ethos" className="footer-link">Ethos</Link></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h3 className="footer-section-title">Community</h3>
            <ul className="footer-links">
              <li><Link href="/write-for-us" className="footer-link">Write for Us</Link></li>
              <li><a href="#" className="footer-link">Discord</a></li>
              <li><a href="#" className="footer-link">Twitter</a></li>
              <li><a href="#" className="footer-link">GitHub</a></li>
            </ul>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; 2026 proindra. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}