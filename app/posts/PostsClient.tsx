'use client';

import { useState, useEffect } from 'react';
import { Search } from 'lucide-react';
import BlogCard from '@/components/BlogCard';

interface PostsClientProps {
  posts: any[];
}

export default function PostsClient({ posts }: PostsClientProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const userMode = sessionStorage.getItem('userMode');
    
    if (!userMode) {
      window.location.href = '/login';
      return;
    }
    
    setIsAuthenticated(true);
    setIsLoading(false);
  }, []);

  if (isLoading) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-header">
            <h1 className="auth-title">Checking Access...</h1>
          </div>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  const filteredPosts = posts.filter(post =>
    post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    post.excerpt.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="page-container">
      <div className="page-hero">
        <h1 className="page-title">All Posts</h1>
        <p className="page-subtitle">
          Dive deep into machine learning research, tutorials, and insights from our community
        </p>
      </div>

      <div className="posts-search">
        <div className="search-container">
          <Search className="search-icon" size={20} />
          <input
            type="text"
            placeholder="Search posts..."
            className="search-input"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {filteredPosts.length > 0 ? (
        <div className="posts-grid">
          {filteredPosts.map((post) => (
            <BlogCard key={post.slug} post={post} />
          ))}
        </div>
      ) : (
        <div className="posts-empty">
          <div className="empty-icon"></div>
          <h3>No posts found</h3>
          <p>Try adjusting your search terms</p>
        </div>
      )}
    </div>
  );
}