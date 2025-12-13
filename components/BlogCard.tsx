import Link from 'next/link';
import Image from 'next/image';
import { Clock, User } from 'lucide-react';
import { BlogPost } from '@/types';

interface BlogCardProps {
  post: BlogPost;
  featured?: boolean;
}

export default function BlogCard({ post, featured = false }: BlogCardProps) {
  return (
    <article className={`blog-card ${featured ? 'featured' : ''}`}>
      <Link href={`/posts/${post.slug}`}>
        <div>
          {post.image && (
            <div className="blog-card-image">
              <Image
                src={post.image}
                alt={post.title}
                fill
                className="image-fill"
                sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
              />
            </div>
          )}
          <div className="blog-card-content">
            <div className="blog-card-meta">
              <div className="blog-card-author">
                <User size={14} />
                <span>{post.author}</span>
              </div>
              <time>
                {new Date(post.date).toLocaleDateString('en-GB', { 
                  day: '2-digit', 
                  month: '2-digit', 
                  year: '2-digit' 
                })}
              </time>
            </div>
            
            <h2 className="blog-card-title">
              {post.title}
            </h2>
            
            <p className="blog-card-excerpt">
              {post.excerpt}
            </p>
            
            <div className="blog-card-footer">
              <div className="blog-card-read-time">
                <Clock size={12} />
                <span>{post.readTime} min</span>
              </div>
              
              {post.tags.length > 0 && (
                <div className="blog-card-tags">
                  {post.tags.slice(0, 2).map((tag) => (
                    <span key={tag} className="blog-tag">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </Link>
    </article>
  );
}