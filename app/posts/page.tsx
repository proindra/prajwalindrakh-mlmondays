import { getAllPosts } from '@/lib/posts';
import BlogCard from '@/components/BlogCard';
import { Search } from 'lucide-react';

export default function PostsPage() {
  const posts = getAllPosts();

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
          />
        </div>
      </div>

      {posts.length > 0 ? (
        <div className="posts-grid">
          {posts.map((post) => (
            <BlogCard key={post.slug} post={post} />
          ))}
        </div>
      ) : (
        <div className="posts-empty">
          <div className="empty-icon"></div>
          <h3>No posts yet</h3>
          <p>Be the first to contribute to ML Mondays!</p>
          <a href="/write-for-us" className="btn-primary">
            Write for Us
          </a>
        </div>
      )}
    </div>
  );
}