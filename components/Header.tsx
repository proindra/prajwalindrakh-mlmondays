'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { Menu, X, Github, Twitter } from 'lucide-react';
import Image from 'next/image';
import { getImagePath } from '@/lib/utils';

export default function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [userMode, setUserMode] = useState<string | null>(null);
  const pathname = usePathname();

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Posts', href: '/posts' },
    { name: 'About', href: '/about' },
    { name: 'Ethos', href: '/ethos' },
  ];

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    setUserMode(sessionStorage.getItem('userMode'));
  }, []);

  const handleLogout = () => {
    sessionStorage.removeItem('userMode');
    sessionStorage.removeItem('guestId');
    setUserMode(null);
    window.location.href = '/prajwalindrakh-mlmondays/';
  };

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  return (
    <header className={`header ${scrolled ? 'header-scrolled' : ''}`}>
      <nav className="header-nav">
        <div className="header-content">
          <Link href="/" className="header-logo">
            <div className="logo-container">
              <Image
                src={getImagePath("/neural-hive-logo.svg")}
                alt="Neural Hive Logo"
                width={64}
                height={64}
                className="logo-image"
              />
            </div>
            <div className="logo-text">
              <span className="logo-title gradient-text animate-pulse-glow">
                Neural Insights
              </span>
              <span className="logo-subtitle">Neural Hive</span>
            </div>
          </Link>

          <div className="desktop-nav">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className={`nav-link ${isActive(item.href) ? 'nav-link-active' : ''}`}
              >
                {item.name}
                {isActive(item.href) && <div className="nav-indicator"></div>}
              </Link>
            ))}
          </div>

          <div className="desktop-actions">
            {userMode === 'guest' ? (
              <button onClick={handleLogout} className="nav-link logout-btn">
                Logout (Guest)
              </button>
            ) : (
              <Link href="/login" className="nav-link">
                Sign In
              </Link>
            )}
            <Link href="/write-for-us" className="cta-button">
              Write for Us
            </Link>
          </div>

          <button className="mobile-menu-button" onClick={() => setIsOpen(!isOpen)}>
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {isOpen && (
          <div className="mobile-nav">
            <div className="mobile-nav-content">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`mobile-nav-link ${isActive(item.href) ? 'mobile-nav-link-active' : ''}`}
                  onClick={() => setIsOpen(false)}
                >
                  {item.name}
                </Link>
              ))}
              
              <div className="mobile-nav-footer">
                <Link href="/write-for-us" className="mobile-cta-button" onClick={() => setIsOpen(false)}>
                  Write for Us
                </Link>
              </div>
            </div>
          </div>
        )}
      </nav>
    </header>
  );
}